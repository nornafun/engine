import { Shader } from '../shader.js';
import { SHADERLANGUAGE_WGSL } from '../constants.js';
import { Debug, DebugHelper } from '../../../core/debug.js';
import { DebugGraphics } from '../debug-graphics.js';
import webgpuMipmap from '../shader-chunks/frag/webgpu-mipmap.js';

/**
 * @import { WebgpuGraphicsDevice } from './webgpu-graphics-device.js'
 * @import { WebgpuShader } from './webgpu-shader.js'
 * @import { WebgpuTexture } from './webgpu-texture.js'
 */

/**
 * A WebGPU helper class implementing texture mipmap generation.
 *
 * Its render pipelines (one per texture format) are created synchronously on first use, even with
 * {@link WebgpuGraphicsDevice#asyncPipelines} on: the mip chain is rendered the moment a texture is
 * uploaded, so a pending pipeline would leave the texture with a single level. A format used in
 * play is warmed instead with {@link WebgpuMipmapRenderer#precompile}, which compiles off thread.
 *
 * @ignore
 */
class WebgpuMipmapRenderer {
    /** @type {WebgpuGraphicsDevice} */
    device;

    /**
     * Cache of render pipelines keyed by texture format.
     *
     * @type {Map<string, GPURenderPipeline>}
     * @private
     */
    pipelineCache = new Map();

    /**
     * The formats precompile is compiling, and so their number is the pending count.
     *
     * @type {Set<string>}
     * @private
     */
    compiling = new Set();

    /**
     * Pipelines created synchronously inside generate, which can stall the frame: the number of
     * texture formats that were not precompiled.
     *
     * @type {number}
     */
    syncCreated = 0;

    constructor(device) {
        this.device = device;

        // shader that renders a fullscreen textured quad
        this.shader = new Shader(device, {
            name: 'WebGPUMipmapRendererShader',
            shaderLanguage: SHADERLANGUAGE_WGSL,
            vshader: webgpuMipmap,
            fshader: webgpuMipmap
        });

        // using minified rendering, so that's the only filter mode we need to set.
        this.minSampler = device.wgpu.createSampler({ minFilter: 'linear' });
    }

    destroy() {
        this.shader.destroy();
        this.shader = null;
        this.pipelineCache = null;
    }

    /**
     * The number of asynchronous compiles started by precompile that have not settled.
     *
     * @type {number}
     */
    get pending() {
        return this.compiling.size;
    }

    /**
     * @param {GPUTextureFormat} format - The texture format.
     * @returns {GPURenderPipelineDescriptor} The descriptor of the pipeline for the format.
     * @private
     */
    getDescriptor(format) {

        /** @type {WebgpuShader} */
        const webgpuShader = this.shader.impl;

        return {
            layout: 'auto',
            vertex: {
                module: webgpuShader.getVertexShaderModule(),
                entryPoint: webgpuShader.vertexEntryPoint
            },
            fragment: {
                module: webgpuShader.getFragmentShaderModule(),
                entryPoint: webgpuShader.fragmentEntryPoint,
                targets: [{
                    format: format
                }]
            },
            primitive: {
                topology: 'triangle-strip'
            }
        };
    }

    /**
     * Compiles the mipmap pipelines of the given texture formats with createRenderPipelineAsync, so
     * that the first texture of each format does not create one synchronously inside generate. A
     * format whose pipeline already exists, or is being compiled, is skipped. Counted by
     * {@link WebgpuGraphicsDevice#pendingPipelines} while compiling.
     *
     * @param {GPUTextureFormat[]} formats - The texture formats, for example 'rgba8unorm',
     * 'rgba8unorm-srgb' and 'rgba16float'.
     * @returns {Promise<void>} Resolves once all the pipelines are compiled. Never rejects: a
     * failed compile leaves the format to be created synchronously by generate.
     */
    precompile(formats) {
        const wgpu = this.device.wgpu;
        const promises = [];
        if (typeof wgpu.createRenderPipelineAsync !== 'function') {
            return Promise.resolve();
        }
        for (const format of formats) {
            if (this.pipelineCache.has(format) || this.compiling.has(format)) continue;

            this.compiling.add(format);
            promises.push(wgpu.createRenderPipelineAsync(this.getDescriptor(format)).then((pipeline) => {
                this.compiling.delete(format);

                // generate may have created it synchronously meanwhile, or the renderer was destroyed
                if (this.pipelineCache && !this.pipelineCache.has(format)) {
                    DebugHelper.setLabel(pipeline, `RenderPipeline-MipmapRenderer-${format}`);
                    this.pipelineCache.set(format, pipeline);
                }
            }, (error) => {
                this.compiling.delete(format);
                Debug.warn(`Asynchronous mipmap pipeline creation failed for ${format}, it is created synchronously instead.`, error);
            }));
        }
        return Promise.all(promises).then(() => {});
    }

    /**
     * Generates mipmaps for the specified WebGPU texture.
     *
     * @param {WebgpuTexture} webgpuTexture - The texture to generate mipmaps for.
     */
    generate(webgpuTexture) {

        // ignore texture with no mipmaps
        const textureDescr = webgpuTexture.desc;
        if (textureDescr.mipLevelCount <= 1) {
            return;
        }

        // not all types are currently supported
        if (webgpuTexture.texture.volume) {
            Debug.warnOnce('WebGPU mipmap generation is not supported volume texture.', webgpuTexture.texture);
            return;
        }

        const device = this.device;
        const wgpu = device.wgpu;
        const format = textureDescr.format;

        // Get or create cached pipeline for this texture format
        let pipeline = this.pipelineCache.get(format);
        if (!pipeline) {
            this.syncCreated++;
            pipeline = wgpu.createRenderPipeline(this.getDescriptor(format));
            DebugHelper.setLabel(pipeline, `RenderPipeline-MipmapRenderer-${format}`);
            this.pipelineCache.set(format, pipeline);
        }

        const texture = webgpuTexture.texture;
        const numFaces = texture.cubemap ? 6 : (texture.array ? texture.arrayLength : 1);

        const srcViews = [];
        for (let face = 0; face < numFaces; face++) {
            srcViews.push(webgpuTexture.createView({
                dimension: '2d',
                baseMipLevel: 0,
                mipLevelCount: 1,
                baseArrayLayer: face
            }));
        }

        // loop through each mip level and render the previous level's contents into it.
        const commandEncoder = device.getCommandEncoder();

        DebugGraphics.pushGpuMarker(device, 'MIPMAP-RENDERER');

        for (let i = 1; i < textureDescr.mipLevelCount; i++) {

            for (let face = 0; face < numFaces; face++) {

                const dstView = webgpuTexture.createView({
                    dimension: '2d',
                    baseMipLevel: i,
                    mipLevelCount: 1,
                    baseArrayLayer: face
                });

                const passEncoder = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: dstView,
                        loadOp: 'clear',
                        storeOp: 'store'
                    }]
                });
                DebugHelper.setLabel(passEncoder, `MipmapRenderer-PassEncoder_${i}`);

                const bindGroup = wgpu.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [{
                        binding: 0,
                        resource: this.minSampler
                    }, {
                        binding: 1,
                        resource: srcViews[face]
                    }]
                });

                passEncoder.setPipeline(pipeline);
                passEncoder.setBindGroup(0, bindGroup);
                passEncoder.draw(4);
                passEncoder.end();

                // next iteration
                srcViews[face] = dstView;
            }
        }

        DebugGraphics.popGpuMarker(device);

        // clear invalidated state
        device.pipeline = null;
    }
}

export { WebgpuMipmapRenderer };
