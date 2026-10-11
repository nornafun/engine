import { expect } from 'chai';

import {
    BINDGROUP_MATERIAL, SEMANTIC_NORMAL, SEMANTIC_POSITION, SEMANTIC_TEXCOORD0, TYPE_FLOAT32
} from '../../../src/platform/graphics/constants.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { VertexFormat } from '../../../src/platform/graphics/vertex-format.js';
import { CameraShaderParams } from '../../../src/scene/camera-shader-params.js';
import {
    LIGHTTYPE_DIRECTIONAL, LIGHTTYPE_OMNI, MASK_AFFECT_DYNAMIC, SHADER_FORWARD, SHADER_PREPASS,
    SHADERDEF_AFFECT_DYNAMIC, SHADERDEF_NOSHADOW
} from '../../../src/scene/constants.js';
import { Light } from '../../../src/scene/light.js';
import { LightList } from '../../../src/scene/lighting/light-list.js';
import { StandardMaterial } from '../../../src/scene/materials/standard-material.js';
import { LitOptionsUtils } from '../../../src/scene/shader-lib/programs/lit-options-utils.js';
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

});
