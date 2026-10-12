import { expect } from 'chai';

import {
    BINDGROUP_MATERIAL, SEMANTIC_NORMAL, SEMANTIC_POSITION, SEMANTIC_TEXCOORD0, TYPE_FLOAT32
} from '../../../src/platform/graphics/constants.js';
import { Shader } from '../../../src/platform/graphics/shader.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { VertexFormat } from '../../../src/platform/graphics/vertex-format.js';
import { CameraShaderParams } from '../../../src/scene/camera-shader-params.js';
import {
    LIGHTTYPE_DIRECTIONAL, LIGHTTYPE_OMNI, MASK_AFFECT_DYNAMIC, SHADER_FORWARD, SHADER_PICK, SHADER_PREPASS,
    SHADERDEF_AFFECT_DYNAMIC, SHADERDEF_NOSHADOW, SHADOW_PCF3_32F
} from '../../../src/scene/constants.js';
import { Light } from '../../../src/scene/light.js';
import { LightList } from '../../../src/scene/lighting/light-list.js';
import { StandardMaterial } from '../../../src/scene/materials/standard-material.js';
import { getProgramLibrary } from '../../../src/scene/shader-lib/get-program-library.js';
import { reachableChunks } from '../../../src/scene/shader-lib/pass-shader-chunks.js';
import { LitOptionsUtils } from '../../../src/scene/shader-lib/programs/lit-options-utils.js';
import { standard } from '../../../src/scene/shader-lib/programs/standard.js';
import { ShaderChunks } from '../../../src/scene/shader-lib/shader-chunks.js';
import { ShaderPass } from '../../../src/scene/shader-pass.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// Shaders whose generated source does not depend on an option or a uniform are shared between the
// materials and mesh instances differing in it alone: the texture transforms of a material only
// split the shaders applying one, and noShadow only the shaders receiving a shadow.
describe('StandardMaterial program sharing', function () {

    let app;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();

        // the clustered lighting shader chunks are registered by the renderer, which does not run
        // in these tests - disable clustered lighting so they are not required
        app.scene.clusteredLightingEnabled = false;
    });

    afterEach(function () {
        app.destroy();
        app = null;
        jsdomTeardown();
    });

    const createTexture = () => new Texture(app.graphicsDevice, { width: 4, height: 4 });

    /**
     * @param {StandardMaterial} material - The material.
     * @param {number} pass - The shader pass.
     * @param {object} [params] - Other parameters of the variant.
     * @returns {import('../../../src/platform/graphics/shader.js').Shader} The shader variant.
     */
    const variant = (material, pass, params = {}) => {
        const device = app.graphicsDevice;
        return material.getShaderVariant({
            device,
            scene: app.scene,
            objDefs: 0,
            pass,
            cameraShaderParams: new CameraShaderParams(),
            viewUniformFormat: app.renderer.getViewUniformFormat(false, new LightList()),
            vertexFormat: new VertexFormat(device, [
                { semantic: SEMANTIC_POSITION, components: 3, type: TYPE_FLOAT32 },
                { semantic: SEMANTIC_NORMAL, components: 3, type: TYPE_FLOAT32 },
                { semantic: SEMANTIC_TEXCOORD0, components: 2, type: TYPE_FLOAT32 }
            ]),
            ...params
        });
    };

    // the shadow pass of a directional light, allocated as the shadow renderer does
    const shadowPass = () => ShaderPass.get(app.graphicsDevice).allocate(`ShadowPass_${LIGHTTYPE_DIRECTIONAL}_${SHADOW_PCF3_32F}`, {
        isShadow: true,
        lightType: LIGHTTYPE_DIRECTIONAL,
        shadowType: SHADOW_PCF3_32F
    }).index;

    const materialFormat = shader => shader.definition.processingOptions.uniformFormats[BINDGROUP_MATERIAL];

    describe('texture transforms', function () {

        /**
         * A material with a diffuse map, and an ao map pointing at the same texture, which shares
         * its sampler: one texture slot, but the transform uniforms of both maps.
         *
         * @param {Texture} texture - The texture.
         * @param {number} aoUv - The uv set of the ao map.
         * @returns {StandardMaterial} The material.
         */
        const createSharing = (texture, aoUv = 0) => {
            const material = new StandardMaterial();
            material.diffuseMap = texture;
            material.aoMap = texture;
            material.aoMapUv = aoUv;
            material.update();
            return material;
        };

        const createDiffuse = (texture) => {
            const material = new StandardMaterial();
            material.diffuseMap = texture;
            material.update();
            return material;
        };

        it('lays the transforms out after the typed properties, which keep their offsets', function () {
            const material = createSharing(createTexture());
            variant(material, SHADER_FORWARD);
            material.prepareForRender(app.graphicsDevice, app.scene);
            const layout = material._layout;
            const full = layout.uniformBufferFormat;
            const required = layout.requiredUniformBufferFormat;

            expect(required).to.not.equal(full);
            expect(required.uniforms.some(uniform => uniform.name.includes('MapTransform'))).to.equal(false);
            expect(full.uniforms.slice(0, required.uniforms.length).map(uniform => [uniform.name, uniform.offset]))
            .to.deep.equal(required.uniforms.map(uniform => [uniform.name, uniform.offset]));
            expect(full.uniforms.slice(required.uniforms.length).map(uniform => uniform.name)).to.have.members([
                'texture_diffuseMapTransform0', 'texture_diffuseMapTransform1',
                'texture_aoMapTransform0', 'texture_aoMapTransform1'
            ]);
            expect(full.byteSize).to.be.greaterThan(required.byteSize);
        });

        it('shares the prepass shader of materials whose layouts differ in the transforms alone', function () {
            const texture = createTexture();
            const sharing = createSharing(texture);
            const diffuse = createDiffuse(texture);
            sharing.prepareForRender(app.graphicsDevice, app.scene);
            diffuse.prepareForRender(app.graphicsDevice, app.scene);

            // one texture slot each, but different uniform buffer layouts
            expect(sharing._layout.bindGroupFormat.key).to.equal(diffuse._layout.bindGroupFormat.key);
            expect(sharing._layout.uniformBufferFormat).to.not.equal(diffuse._layout.uniformBufferFormat);

            const prepass = variant(sharing, SHADER_PREPASS);
            expect(variant(diffuse, SHADER_PREPASS)).to.equal(prepass);
            expect(materialFormat(prepass)).to.equal(sharing._layout.requiredUniformBufferFormat);

            // the forward pass samples the ao map too, so it is a different shader
            expect(variant(diffuse, SHADER_FORWARD)).to.not.equal(variant(sharing, SHADER_FORWARD));
        });

        it('shares the forward shader when the maps are at the identity transform', function () {
            const texture = createTexture();

            // the ao map samples a uv set the mesh does not provide, so only the diffuse map is
            // sampled, as in the other material
            const sharing = createSharing(texture, 1);
            const diffuse = createDiffuse(texture);

            const forward = variant(sharing, SHADER_FORWARD);
            expect(variant(diffuse, SHADER_FORWARD)).to.equal(forward);
            expect(forward.definition.vshader).to.not.include('MapTransform');
        });

        it('processes a shader applying a transform against the whole layout', function () {
            const texture = createTexture();
            const tiled = createDiffuse(texture);
            tiled.diffuseMapTiling.set(4, 4);
            tiled.update();
            tiled.prepareForRender(app.graphicsDevice, app.scene);

            const forward = variant(tiled, SHADER_FORWARD);
            expect(forward.definition.vshader).to.include('texture_diffuseMapTransform');
            expect(materialFormat(forward)).to.equal(tiled._layout.uniformBufferFormat);
            expect(materialFormat(forward).get('texture_diffuseMapTransform0')).to.exist;

            // the prepass samples no map without an alpha test
            expect(materialFormat(variant(tiled, SHADER_PREPASS))).to.equal(tiled._layout.requiredUniformBufferFormat);
        });

        it('processes the prepass of an alpha tested map with a transform against the whole layout', function () {
            const material = new StandardMaterial();
            material.opacityMap = createTexture();
            material.opacityMapTiling.set(2, 2);
            material.alphaTest = 0.5;
            material.update();
            material.prepareForRender(app.graphicsDevice, app.scene);

            const prepass = variant(material, SHADER_PREPASS);
            expect(prepass.definition.vshader).to.include('texture_opacityMapTransform');
            expect(materialFormat(prepass)).to.equal(material._layout.uniformBufferFormat);
        });

    });

    describe('noShadow', function () {

        /**
         * @param {boolean} castShadows - Whether the light casts shadows.
         * @param {number} [type] - The light type.
         * @returns {LightList} A light list holding one light.
         */
        const lightList = (castShadows, type = LIGHTTYPE_DIRECTIONAL) => {
            const light = new Light(app.graphicsDevice, false);
            light.type = type;
            light.mask = MASK_AFFECT_DYNAMIC;
            light.castShadows = castShadows;
            light.enabled = true;
            const list = new LightList();
            list.update([light], false);
            return list;
        };

        const forward = (material, list, noShadow) => {
            return variant(material, SHADER_FORWARD, {
                lightList: list,
                // a mesh instance lit by the dynamic lights
                objDefs: SHADERDEF_AFFECT_DYNAMIC | (noShadow ? SHADERDEF_NOSHADOW : 0)
            });
        };

        it('does not split the shaders of a mesh lit by no light casting shadows', function () {
            const material = new StandardMaterial();
            material.update();
            const list = lightList(false);
            expect(forward(material, list, true)).to.equal(forward(material, list, false));
        });

        it('does not split the shaders of an unlit material', function () {
            const material = new StandardMaterial();
            material.useLighting = false;
            material.update();
            const list = lightList(true);
            expect(forward(material, list, true)).to.equal(forward(material, list, false));
        });

        it('splits the shaders of a mesh lit by a light casting shadows', function () {
            const material = new StandardMaterial();
            material.update();
            const list = lightList(true);
            const receiving = forward(material, list, false);
            const notReceiving = forward(material, list, true);
            expect(notReceiving).to.not.equal(receiving);
            expect(receiving.definition.fshader).to.not.equal(notReceiving.definition.fshader);
        });

        it('keys noShadow only where a shadow is read', function () {
            const caster = { _type: LIGHTTYPE_DIRECTIONAL, castShadows: true, key: 1 };
            const omniCaster = { _type: LIGHTTYPE_OMNI, castShadows: true, key: 2 };
            const plain = { _type: LIGHTTYPE_DIRECTIONAL, castShadows: false, key: 3 };
            const key = options => LitOptionsUtils.generateKey(options);
            const pair = (options) => {
                return [key({ ...options, noShadow: true }), key({ ...options, noShadow: false })];
            };
            const splits = (options) => {
                const [a, b] = pair(options);
                return a !== b;
            };

            expect(splits({ lights: [] })).to.equal(false);
            expect(splits({ lights: [plain] })).to.equal(false);
            expect(splits({ lights: [undefined, caster] })).to.equal(true);

            // with clustered lighting, the local lights are evaluated by the clusters
            expect(splits({ lights: [omniCaster] })).to.equal(true);
            expect(splits({ lights: [omniCaster], clusteredLightingEnabled: true })).to.equal(false);
            expect(splits({ lights: [], clusteredLightingEnabled: true, clusteredLightingShadowsEnabled: true })).to.equal(true);
            expect(splits({ lights: [], clusteredLightingEnabled: false, clusteredLightingShadowsEnabled: true })).to.equal(false);
        });

    });

    describe('shader chunks', function () {

        // a chunk the depth passes include, and two only the forward pass includes
        const transformVS = chunk => chunk.replace('vec4 getPosition() {', 'vec4 getPosition() {\n    float movedMarker = 1.0;');
        const forwardChunks = {
            diffusePS: 'uniform vec3 material_diffuse;\nvoid getAlbedo() { dAlbedo = material_diffuse; }\n',
            ambientPS: 'void addAmbient(vec3 worldNormal) { dDiffuseLight += vec3(0.5); }\n'
        };

        const engineChunk = name => ShaderChunks.get(app.graphicsDevice).get(name);

        /**
         * @param {object} glsl - The GLSL chunks of the material.
         * @param {string} [version] - The chunk API version.
         * @returns {StandardMaterial} The material.
         */
        const create = (glsl = {}, version) => {
            const material = new StandardMaterial();
            for (const [name, code] of Object.entries(glsl)) {
                material.getShaderChunks('glsl').set(name, code);
            }
            if (version !== undefined) {
                material.shaderChunksVersion = version;
            }
            material.update();
            return material;
        };

        /**
         * Returns the shader the generator makes from every chunk of the material, as no chunk was
         * left out: the variant is requested with the program library generating, next to it, the
         * shader of the options it was asked for with the material's own chunks.
         *
         * @param {StandardMaterial} material - The material.
         * @param {number} pass - The shader pass.
         * @returns {{ shared: Shader, full: Shader }} The shader of the variant and the full one.
         */
        const withFullChunks = (material, pass) => {
            const device = app.graphicsDevice;
            const library = getProgramLibrary(device);
            const getProgram = library.getProgram;
            let full = null;
            library.getProgram = function (name, options, processingOptions, userMaterialId) {
                const filtered = options.litOptions.shaderChunks;
                options.litOptions.shaderChunks = material.shaderChunks;
                const definition = standard.createShaderDefinition(device, options);
                options.litOptions.shaderChunks = filtered;
                full = new Shader(device, { ...definition, name: 'full' });
                return getProgram.call(this, name, options, processingOptions, userMaterialId);
            };
            try {
                const shared = variant(material, pass);
                return { shared, full };
            } finally {
                library.getProgram = getProgram;
            }
        };

        const source = shader => [shader.definition.vshader, shader.definition.fshader];

        it('shares the depth shaders of materials differing in the forward chunks alone', function () {
            const moved = { transformVS: transformVS(engineChunk('transformVS')) };
            const plain = create(moved);
            const forward = create({ ...moved, ...forwardChunks }, '2.24');

            for (const pass of [SHADER_PREPASS, shadowPass(), SHADER_PICK]) {
                const { shared, full } = withFullChunks(forward, pass);

                // generated from all its chunks, the shader is the one generated without the
                // forward ones, so leaving them out of the key changes no source
                expect(source(full)).to.deep.equal(source(shared));
                expect(full.definition.attributes).to.deep.equal(shared.definition.attributes);
                expect(variant(plain, pass)).to.equal(shared);
                expect(shared.definition.vshader).to.include('movedMarker');
            }

            // the forward pass includes them
            const forwardShader = variant(forward, SHADER_FORWARD);
            expect(forwardShader).to.not.equal(variant(plain, SHADER_FORWARD));
            expect(forwardShader.definition.fshader).to.include('vec3(0.5)');
        });

        it('splits the depth shaders of materials differing in a chunk the pass includes', function () {
            const plain = create();
            const moved = create({ transformVS: transformVS(engineChunk('transformVS')) });
            const prepass = variant(moved, SHADER_PREPASS);
            expect(prepass).to.not.equal(variant(plain, SHADER_PREPASS));
            expect(prepass.definition.vshader).to.include('movedMarker');
        });

        it('keeps the chunks of a material writing a pass define', function () {
            const plain = create();
            const defining = create({ ...forwardChunks, litUserDeclarationPS: '#define FORWARD_PASS\n' });
            expect(variant(defining, SHADER_PREPASS)).to.not.equal(variant(plain, SHADER_PREPASS));
        });

        it('shares the shaders of materials differing in the chunk API version alone', function () {
            const none = create();
            const versioned = create({}, '2.24');
            for (const pass of [SHADER_FORWARD, SHADER_PREPASS]) {
                expect(variant(versioned, pass)).to.equal(variant(none, pass));
            }
            const a = create(forwardChunks);
            const b = create(forwardChunks, '2.24');
            expect(variant(b, SHADER_FORWARD)).to.equal(variant(a, SHADER_FORWARD));
        });

        it('reaches the chunks of the branch the pass defines select', function () {
            const chunkOf = name => engineChunk(name);
            const prepass = reachableChunks(chunkOf, new Set(['PREPASS_PASS']));
            const shadow = reachableChunks(chunkOf, new Set(['SHADOW_PASS']));
            const forward = reachableChunks(chunkOf, new Set(['FORWARD_PASS']));

            expect(prepass.has('litOtherMainPS')).to.equal(true);
            expect(prepass.has('transformVS')).to.equal(true);
            expect(prepass.has('stdFrontEndPS')).to.equal(true);
            expect(prepass.has('litForwardMainPS')).to.equal(false);
            expect(prepass.has('diffusePS')).to.equal(false);
            expect(shadow.has('litShadowMainPS')).to.equal(true);
            expect(shadow.has('litOtherMainPS')).to.equal(false);
            expect(forward.has('litForwardMainPS')).to.equal(true);
            expect(forward.has('diffusePS')).to.equal(true);
            expect(forward.has('ambientPS')).to.equal(true);

            // an unknown condition keeps both branches, and an unclosed block keeps every chunk
            const custom = new Map([
                ['litMainPS', '#ifdef PREPASS_PASS\n#include "a"\n#elif defined(OTHER)\n#include "b"\n#else\n#include "c"\n#endif\n#if defined(OTHER) || !defined(SHADOW_PASS)\n#include "d"\n#endif'],
                ['litMainVS', '']
            ]);
            const reached = reachableChunks(name => custom.get(name) ?? '', new Set(['PREPASS_PASS']));
            expect([...reached].filter(name => name.length === 1).sort()).to.deep.equal(['a', 'd']);
            const shadowReached = reachableChunks(name => custom.get(name) ?? '', new Set(['SHADOW_PASS']));
            expect([...shadowReached].filter(name => name.length === 1).sort()).to.deep.equal(['b', 'c', 'd']);
            custom.set('litMainVS', '#ifdef PREPASS_PASS\n');
            expect(reachableChunks(name => custom.get(name) ?? '', new Set(['PREPASS_PASS']))).to.equal(null);
        });

    });

    describe('clustered lighting', function () {

        it('shares the shadow shaders of a lit and an unlit material', function () {
            app.destroy();
            app = createApp();
            expect(app.scene.clusteredLightingEnabled).to.equal(true);

            const lit = new StandardMaterial();
            lit.update();
            const unlit = new StandardMaterial();
            unlit.useLighting = false;
            unlit.update();
            const shadow = shadowPass();
            expect(variant(lit, shadow)).to.equal(variant(unlit, shadow));

            // the prepass of the lit material writes the normal the clustered lighting asks for
            expect(variant(lit, SHADER_PREPASS)).to.not.equal(variant(unlit, SHADER_PREPASS));
        });

    });

});
