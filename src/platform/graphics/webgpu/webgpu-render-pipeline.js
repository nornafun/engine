import { Debug, DebugHelper } from '../../../core/debug.js';
import { hash32Fnv1a } from '../../../core/hash.js';
import { TRACEID_RENDERPIPELINE_ALLOC } from '../../../core/constants.js';
import { WebgpuVertexBufferLayout } from './webgpu-vertex-buffer-layout.js';
import { WebgpuDebug } from './webgpu-debug.js';
import { WebgpuPipeline } from './webgpu-pipeline.js';
import { DebugGraphics } from '../debug-graphics.js';
import { BlendState } from '../blend-state.js';
import { bindGroupNames, PRIMITIVE_LINESTRIP, PRIMITIVE_TRISTRIP } from '../constants.js';

/**
 * @import { BindGroupFormat } from '../bind-group-format.js'
 * @import { DepthState } from '../depth-state.js'
 * @import { RenderTarget } from '../render-target.js'
 * @import { Shader } from '../shader.js'
 * @import { StencilParameters } from '../stencil-parameters.js'
 * @import { VertexFormat } from '../vertex-format.js'
 * @import { WebgpuShader } from './webgpu-shader.js'
 */

let _pipelineId = 0;

// reused destination for BlendState#getAttachment, to avoid allocations
const _attachmentBlendState = new BlendState();

// GPUTextureFormats which are blendable and have an alpha channel, which the WebGPU spec requires
// of the first color attachment when alpha-to-coverage is enabled. Note that 'rgba32float' also
// qualifies, but only when the float32-blendable feature is available, and so it is handled
// separately.
const _alphaToCoverageFormats = new Set([
    'rgba8unorm',
    'rgba8unorm-srgb',
    'bgra8unorm',
    'bgra8unorm-srgb',
    'rgb10a2unorm',
    'rgba16float'
]);

// Assembles the WebGPU color write mask of the supplied blend state.
const getWriteMask = (blendState) => {
    let writeMask = 0;
    if (blendState.redWrite) writeMask |= GPUColorWrite.RED;
    if (blendState.greenWrite) writeMask |= GPUColorWrite.GREEN;
    if (blendState.blueWrite) writeMask |= GPUColorWrite.BLUE;
    if (blendState.alphaWrite) writeMask |= GPUColorWrite.ALPHA;
    return writeMask;
};

const _primitiveTopology = [
    'point-list',       // PRIMITIVE_POINTS
    'line-list',        // PRIMITIVE_LINES
    undefined,          // PRIMITIVE_LINELOOP
    'line-strip',       // PRIMITIVE_LINESTRIP
    'triangle-list',    // PRIMITIVE_TRIANGLES
    'triangle-strip',   // PRIMITIVE_TRISTRIP
    undefined           // PRIMITIVE_TRIFAN
];

// WebGPU applies a depth bias only to triangles, and requires it to be zero for other topologies
const _usesDepthBias = topology => topology === 'triangle-list' || topology === 'triangle-strip';

const _blendOperation = [
    'add',              // BLENDEQUATION_ADD
    'subtract',         // BLENDEQUATION_SUBTRACT
    'reverse-subtract', // BLENDEQUATION_REVERSE_SUBTRACT
    'min',              // BLENDEQUATION_MIN
    'max'               // BLENDEQUATION_MAX
];

const _blendFactor = [
    'zero',                 // BLENDMODE_ZERO
    'one',                  // BLENDMODE_ONE
    'src',                  // BLENDMODE_SRC_COLOR
    'one-minus-src',        // BLENDMODE_ONE_MINUS_SRC_COLOR
    'dst',                  // BLENDMODE_DST_COLOR
    'one-minus-dst',        // BLENDMODE_ONE_MINUS_DST_COLOR
    'src-alpha',            // BLENDMODE_SRC_ALPHA
    'src-alpha-saturated',  // BLENDMODE_SRC_ALPHA_SATURATE
    'one-minus-src-alpha',  // BLENDMODE_ONE_MINUS_SRC_ALPHA
    'dst-alpha',            // BLENDMODE_DST_ALPHA
    'one-minus-dst-alpha',  // BLENDMODE_ONE_MINUS_DST_ALPHA
    'constant',             // BLENDMODE_CONSTANT
    'one-minus-constant',   // BLENDMODE_ONE_MINUS_CONSTANT
    'src1',                 // BLENDMODE_SRC1_COLOR
    'one-minus-src1',       // BLENDMODE_ONE_MINUS_SRC1_COLOR
    'src1-alpha',           // BLENDMODE_SRC1_ALPHA
    'one-minus-src1-alpha'  // BLENDMODE_ONE_MINUS_SRC1_ALPHA
];

const _compareFunction = [
    'never',                // FUNC_NEVER
    'less',                 // FUNC_LESS
    'equal',                // FUNC_EQUAL
    'less-equal',           // FUNC_LESSEQUAL
    'greater',              // FUNC_GREATER
    'not-equal',            // FUNC_NOTEQUAL
    'greater-equal',        // FUNC_GREATEREQUAL
    'always'                // FUNC_ALWAYS
];

const _cullModes = [
    'none',                 // CULLFACE_NONE
    'back',                 // CULLFACE_BACK
    'front'                 // CULLFACE_FRONT
];

const _frontFace = [
    'ccw',                  // FRONTFACE_CCW
    'cw'                    // FRONTFACE_CW
];

const _stencilOps = [
    'keep',                 // STENCILOP_KEEP
    'zero',                 // STENCILOP_ZERO
    'replace',              // STENCILOP_REPLACE
    'increment-clamp',      // STENCILOP_INCREMENT
    'increment-wrap',       // STENCILOP_INCREMENTWRAP
    'decrement-clamp',      // STENCILOP_DECREMENT
    'decrement-wrap',       // STENCILOP_DECREMENTWRAP
    'invert'                // STENCILOP_INVERT
];

const _indexFormat = [
    '',                     // INDEXFORMAT_UINT8
    'uint16',               // INDEXFORMAT_UINT16
    'uint32'                // INDEXFORMAT_UINT32
];

class CacheEntry {
    /**
     * Render pipeline. Null while an asynchronous compile of it is pending (see
     * {@link WebgpuRenderPipeline#pending}), or after that compile failed.
     *
     * @type {GPURenderPipeline|null}
     * @private
     */
    pipeline;

    /**
     * True while createRenderPipelineAsync of this entry has not settled.
     *
     * @type {boolean}
     */
    compiling = false;

    /**
     * True when the asynchronous compile of this entry failed, so it is created synchronously by
     * the next lookup, where the usual validation reports the error.
     *
     * @type {boolean}
     */
    failed = false;

    /**
     * The full array of hashes used to lookup the pipeline, used in case of hash collision.
     *
     * @type {Uint32Array}
     */
    hashes;
}

class WebgpuRenderPipeline extends WebgpuPipeline {
    lookupHashes = new Uint32Array(20);

    // a float view of the lookup hashes, to store the float values by their bits
    lookupHashesFloat = new Float32Array(this.lookupHashes.buffer);

    constructor(device) {
        super(device);

        /**
         * The cache of vertex buffer layouts
         *
         * @type {WebgpuVertexBufferLayout}
         */
        this.vertexBufferLayout = new WebgpuVertexBufferLayout();

        /**
         * The cache of render pipelines
         *
         * @type {Map<number, CacheEntry[]>}
         */
        this.cache = new Map();

        /**
         * The number of asynchronous compiles in flight, see {@link WebgpuGraphicsDevice#pendingPipelines}.
         *
         * @type {number}
         */
        this.pending = 0;

        /**
         * Pipelines created by this class, by how: `sync` inside a lookup (can stall the frame),
         * `async` through createRenderPipelineAsync.
         */
        this.created = { sync: 0, async: 0 };

        /**
         * Resolvers of the promises handed out by {@link WebgpuRenderPipeline#whenIdle}.
         *
         * @type {Function[]}
         * @private
         */
        this._idleWaiters = [];

        /**
         * Bumped by clearCache, so that a compile started before it can tell it is stale.
         *
         * @type {number}
         * @private
         */
        this._generation = 0;
    }

    /**
     * Drops every cached pipeline, on a device loss: the pipelines belong to the lost native
     * device, and its compiles in flight never count again.
     */
    clearCache() {
        this.cache.clear();
        this._generation++;
        this.pending = 0;
        this._flushIdleWaiters();
    }

    /** @private */
    _flushIdleWaiters() {
        const waiters = this._idleWaiters;
        this._idleWaiters = [];
        waiters.forEach(resolve => resolve());
    }

    /**
     * Returns a promise that resolves once no asynchronous compile is in flight - immediately
     * when there is none.
     *
     * @returns {Promise<void>} The promise.
     */
    whenIdle() {
        if (this.pending === 0) {
            return Promise.resolve();
        }
        return new Promise((resolve) => {
            this._idleWaiters.push(resolve);
        });
    }

    /**
     * @param {CacheEntry} entry - The entry whose asynchronous compile settled.
     * @param {number} generation - The generation the compile started in.
     * @private
     */
    _settled(entry, generation) {
        entry.compiling = false;
        if (generation === this._generation && --this.pending === 0) {
            this._flushIdleWaiters();
        }
    }

    /**
     * Returns the index format a render pipeline depends on. Only a strip topology uses it, as
     * the strip index format of the pipeline - for any other topology it takes no part, so that
     * meshes of 16 and 32 bit indices share a pipeline.
     *
     * @param {number} primitiveType - The primitive type.
     * @param {number|undefined} ibFormat - The index buffer format.
     * @returns {number|undefined} The index format for a strip topology, undefined otherwise.
     */
    static stripIndexFormat(primitiveType, ibFormat) {
        return (primitiveType === PRIMITIVE_LINESTRIP || primitiveType === PRIMITIVE_TRISTRIP) ? ibFormat : undefined;
    }

    /**
     * @param {object} primitive - The primitive.
     * @param {VertexFormat} vertexFormat0 - The first vertex format.
     * @param {VertexFormat} vertexFormat1 - The second vertex format.
     * @param {number|undefined} ibFormat - The index buffer format.
     * @param {Shader} shader - The shader.
     * @param {RenderTarget} renderTarget - The render target.
     * @param {BindGroupFormat[]} bindGroupFormats - An array of bind group formats.
     * @param {BlendState} blendState - The blend state.
     * @param {DepthState} depthState - The depth state.
     * @param {number} cullMode - The cull mode.
     * @param {boolean} stencilEnabled - Whether stencil is enabled.
     * @param {StencilParameters} stencilFront - The stencil state for front faces.
     * @param {StencilParameters} stencilBack - The stencil state for back faces.
     * @param {number} frontFace - The front face.
     * @param {boolean} alphaToCoverage - Whether alpha to coverage is requested.
     * @returns {GPURenderPipeline|null} Returns the render pipeline. Null only when
     * {@link WebgpuGraphicsDevice#asyncPipelines} is on and the pipeline is still compiling: the
     * lookup has started or is awaiting the compile, and the caller must skip its draw and look
     * the pipeline up again later. With the flag off it is never null.
     * @private
     */
    get(primitive, vertexFormat0, vertexFormat1, ibFormat, shader, renderTarget, bindGroupFormats, blendState,
        depthState, cullMode, stencilEnabled, stencilFront, stencilBack, frontFace, alphaToCoverage) {

        Debug.assert(bindGroupFormats.length <= bindGroupNames.length);

        // ibFormat is used only for stripped primitives, clear it otherwise to avoid additional render pipelines
        const primitiveType = primitive.type;
        ibFormat = WebgpuRenderPipeline.stripIndexFormat(primitiveType, ibFormat);

        // all bind groups must be set as the WebGPU layout cannot have skipped indices. Not having a bind
        // group would assign incorrect slots to the following bind groups, causing a validation errors.
        Debug.call(() => {
            for (let i = 0; i < bindGroupNames.length; i++) {
                Debug.assert(bindGroupFormats[i], `BindGroup with index ${i} [${bindGroupNames[i]}] is not set.`);
            }
        });

        // alpha to coverage is dropped when the render target cannot support it, so the effective
        // state is what needs to take part in the hash
        const alphaToCoverageEnabled = this.getAlphaToCoverage(alphaToCoverage, renderTarget);

        // the depth bias takes part in the hash as WebGPU applies it - only to triangles, and with
        // its constant part truncated to an integer - so that the depth states differing only in
        // what WebGPU ignores share a pipeline
        const primitiveTopology = _primitiveTopology[primitiveType];
        const usesDepthBias = _usesDepthBias(primitiveTopology);

        // render pipeline unique hash
        const { lookupHashes, lookupHashesFloat } = this;
        lookupHashes[0] = primitiveType;
        lookupHashes[1] = shader.id;
        lookupHashes[2] = cullMode;
        lookupHashes[3] = depthState.func;
        lookupHashes[4] = blendState.key;
        lookupHashes[5] = vertexFormat0?.renderingHash ?? 0;
        lookupHashes[6] = vertexFormat1?.renderingHash ?? 0;
        lookupHashes[7] = renderTarget.impl.key;
        lookupHashes[8] = bindGroupFormats[0]?.key ?? 0;
        lookupHashes[9] = bindGroupFormats[1]?.key ?? 0;
        lookupHashes[10] = bindGroupFormats[2]?.key ?? 0;
        lookupHashes[11] = bindGroupFormats[3]?.key ?? 0;
        lookupHashes[12] = stencilEnabled ? stencilFront.key : 0;
        lookupHashes[13] = stencilEnabled ? stencilBack.key : 0;
        lookupHashes[14] = ibFormat ?? 0;
        lookupHashes[15] = frontFace;
        lookupHashes[16] = alphaToCoverageEnabled ? 1 : 0;
        lookupHashes[17] = depthState.write ? 1 : 0;
        lookupHashes[18] = usesDepthBias ? Math.trunc(depthState.depthBias) : 0;
        lookupHashesFloat[19] = usesDepthBias ? depthState.depthBiasSlope : 0;
        const hash = hash32Fnv1a(lookupHashes);

        // cached pipeline
        let cacheEntries = this.cache.get(hash);

        // an entry of this key without a pipeline, to be created now (see below)
        let cacheEntry = null;
        const useAsync = this.device.asyncPipelines && typeof this.device.wgpu?.createRenderPipelineAsync === 'function';

        // if we have cache entries, find the exact match, as hash collision can occur
        if (cacheEntries) {
            for (let i = 0; i < cacheEntries.length; i++) {
                const entry = cacheEntries[i];
                if (WebgpuPipeline.keysEqual(entry.hashes, lookupHashes)) {
                    if (entry.pipeline) {
                        return entry.pipeline;
                    }

                    // compiling: still pending while asynchronous pipelines stay on, otherwise the
                    // caller needs the pipeline now - create it synchronously, and the late result
                    // of the compile is dropped. A failed compile is created synchronously too.
                    if (entry.compiling && useAsync) {
                        return null;
                    }
                    cacheEntry = entry;
                    break;
                }
            }
        }

        // no match or a hash collision, so create a new pipeline
        Debug.assert(primitiveTopology, 'Unsupported primitive topology', primitive);

        // pipeline layout
        const pipelineLayout = this.getPipelineLayout(bindGroupFormats);

        // vertex buffer layout
        const vertexBufferLayout = this.vertexBufferLayout.get(vertexFormat0, vertexFormat1);

        // pipeline
        const isNew = !cacheEntry;
        if (isNew) {
            cacheEntry = new CacheEntry();
            cacheEntry.hashes = new Uint32Array(lookupHashes);
            cacheEntry.pipeline = null;
        }

        if (useAsync && !cacheEntry.failed) {

            // the entry takes its place in the cache now, with no pipeline until the compile
            // settles, so the same key is never compiled twice
            this.createAsync(cacheEntry, primitiveTopology, ibFormat, shader, renderTarget, pipelineLayout, blendState,
                depthState, vertexBufferLayout, cullMode, stencilEnabled, stencilFront, stencilBack, frontFace,
                alphaToCoverageEnabled);
        } else {
            cacheEntry.pipeline = this.create(primitiveTopology, ibFormat, shader, renderTarget, pipelineLayout, blendState,
                depthState, vertexBufferLayout, cullMode, stencilEnabled, stencilFront, stencilBack, frontFace,
                alphaToCoverageEnabled);
        }

        // add to cache
        if (isNew) {
            if (cacheEntries) {
                cacheEntries.push(cacheEntry);
            } else {
                cacheEntries = [cacheEntry];
            }
            this.cache.set(hash, cacheEntries);
        }

        return cacheEntry.pipeline;
    }

    getBlend(blendState) {

        // blend needs to be undefined when blending is disabled
        let blend;

        if (blendState.blend) {

            /** @type {GPUBlendState} */
            blend = {
                color: {
                    operation: _blendOperation[blendState.colorOp],
                    srcFactor: _blendFactor[blendState.colorSrcFactor],
                    dstFactor: _blendFactor[blendState.colorDstFactor]
                },
                alpha: {
                    operation: _blendOperation[blendState.alphaOp],
                    srcFactor: _blendFactor[blendState.alphaSrcFactor],
                    dstFactor: _blendFactor[blendState.alphaDstFactor]
                }
            };

            // unsupported blend factors
            Debug.assert(blend.color.srcFactor !== undefined);
            Debug.assert(blend.color.dstFactor !== undefined);
            Debug.assert(blend.alpha.srcFactor !== undefined);
            Debug.assert(blend.alpha.dstFactor !== undefined);
        }

        return blend;
    }

    /**
     * Alpha to coverage is part of the immutable pipeline state on WebGPU, and the spec only allows
     * it when the render target is multi-sampled and its first color attachment uses a blendable
     * format with an alpha channel. A material is not tied to a single render target - the same one
     * can be rendered into a multi-sampled forward pass, a single-sampled pass, or a depth-only
     * shadow pass with no color attachment at all - so the flag is dropped where it cannot be used
     * instead of failing the pipeline creation. This matches WebGL, where enabling
     * SAMPLE_ALPHA_TO_COVERAGE on a single-sampled framebuffer is a no-op rather than an error.
     *
     * @param {boolean} alphaToCoverage - The requested alpha to coverage state.
     * @param {RenderTarget} renderTarget - The render target.
     * @returns {boolean} Returns true if alpha to coverage can be enabled for the render target.
     * @private
     */
    getAlphaToCoverage(alphaToCoverage, renderTarget) {

        // requires a multi-sampled target - this also covers depth-only passes, which have no
        // color attachments and are never multi-sampled
        if (!alphaToCoverage || renderTarget.samples <= 1) {
            return false;
        }

        const format = renderTarget.impl.colorAttachments[0]?.format;
        const supported = _alphaToCoverageFormats.has(format) ||
            (format === 'rgba32float' && this.device.textureFloatBlendable);

        // this case is worth reporting - alpha to coverage was asked for on a multi-sampled target,
        // and the only reason it cannot be honored is the format of the first color attachment
        if (!supported) {
            Debug.warnOnce('Alpha to coverage is ignored, as it requires the first color attachment to use a blendable format with an alpha channel. Format:', format);
        }

        return supported;
    }

    /**
     * @param {DepthState} depthState - The depth state.
     * @param {RenderTarget} renderTarget - The render target.
     * @param {boolean} stencilEnabled - Whether stencil is enabled.
     * @param {StencilParameters} stencilFront - The stencil state for front faces.
     * @param {StencilParameters} stencilBack - The stencil state for back faces.
     * @param {string} primitiveTopology - The primitive topology.
     * @returns {object} Returns the depth stencil state.
     * @private
     */
    getDepthStencil(depthState, renderTarget, stencilEnabled, stencilFront, stencilBack, primitiveTopology) {

        /** @type {GPUDepthStencilState} */
        let depthStencil;
        const { depth, stencil } = renderTarget;
        if (depth || stencil) {

            // format of depth-stencil attachment
            depthStencil = {
                format: renderTarget.impl.depthAttachment.format
            };

            // depth
            if (depth) {
                depthStencil.depthWriteEnabled = depthState.write;
                depthStencil.depthCompare = _compareFunction[depthState.func];

                // GPUDepthBias is an integer, which the pipeline hash relies on as well
                const biasAllowed = _usesDepthBias(primitiveTopology);
                depthStencil.depthBias = biasAllowed ? Math.trunc(depthState.depthBias) : 0;
                depthStencil.depthBiasSlopeScale = biasAllowed ? depthState.depthBiasSlope : 0;
            } else {
                // if render target does not have depth buffer
                depthStencil.depthWriteEnabled = false;
                depthStencil.depthCompare = 'always';
            }

            // stencil
            if (stencil && stencilEnabled) {

                // Note that WebGPU only supports a single mask, we use the one from front, but not from back.
                depthStencil.stencilReadMask = stencilFront.readMask;
                depthStencil.stencilWriteMask = stencilFront.writeMask;

                depthStencil.stencilFront = {
                    compare: _compareFunction[stencilFront.func],
                    failOp: _stencilOps[stencilFront.fail],
                    passOp: _stencilOps[stencilFront.zpass],
                    depthFailOp: _stencilOps[stencilFront.zfail]
                };

                depthStencil.stencilBack = {
                    compare: _compareFunction[stencilBack.func],
                    failOp: _stencilOps[stencilBack.fail],
                    passOp: _stencilOps[stencilBack.zpass],
                    depthFailOp: _stencilOps[stencilBack.zfail]
                };
            }
        }

        return depthStencil;
    }

    /**
     * Builds the descriptor of a render pipeline, shared by the synchronous and the asynchronous
     * creation.
     *
     * @param {string} primitiveTopology - The primitive topology.
     * @param {number|undefined} ibFormat - The strip index buffer format.
     * @param {Shader} shader - The shader.
     * @param {RenderTarget} renderTarget - The render target.
     * @param {GPUPipelineLayout|string} pipelineLayout - The pipeline layout.
     * @param {BlendState} blendState - The blend state.
     * @param {DepthState} depthState - The depth state.
     * @param {GPUVertexBufferLayout[]} vertexBufferLayout - The vertex buffer layouts.
     * @param {number} cullMode - The cull mode.
     * @param {boolean} stencilEnabled - Whether stencil is enabled.
     * @param {StencilParameters} stencilFront - The stencil state for front faces.
     * @param {StencilParameters} stencilBack - The stencil state for back faces.
     * @param {number} frontFace - The front face.
     * @param {boolean} alphaToCoverageEnabled - Whether alpha to coverage is enabled.
     * @returns {GPURenderPipelineDescriptor} The descriptor.
     * @private
     */
    buildDescriptor(primitiveTopology, ibFormat, shader, renderTarget, pipelineLayout, blendState, depthState,
        vertexBufferLayout, cullMode, stencilEnabled, stencilFront, stencilBack, frontFace, alphaToCoverageEnabled) {

        /** @type {WebgpuShader} */
        const webgpuShader = shader.impl;

        /** @type {GPURenderPipelineDescriptor} */
        const desc = {
            vertex: {
                module: webgpuShader.getVertexShaderModule(),
                entryPoint: webgpuShader.vertexEntryPoint,
                buffers: vertexBufferLayout
            },

            primitive: {
                topology: primitiveTopology,
                frontFace: _frontFace[frontFace],
                cullMode: _cullModes[cullMode]
            },

            depthStencil: this.getDepthStencil(depthState, renderTarget, stencilEnabled, stencilFront, stencilBack, primitiveTopology),

            multisample: {
                count: renderTarget.samples,
                alphaToCoverageEnabled: alphaToCoverageEnabled
            },

            // uniform / texture binding layout
            layout: pipelineLayout
        };

        if (ibFormat) {
            desc.primitive.stripIndexFormat = _indexFormat[ibFormat];
        }

        desc.fragment = {
            module: webgpuShader.getFragmentShaderModule(),
            entryPoint: webgpuShader.fragmentEntryPoint,
            targets: []
        };

        const colorAttachments = renderTarget.impl.colorAttachments;
        if (blendState.usesDualSourceBlending) {
            Debug.assert(shader.definition.useDualSourceBlending,
                'A BlendState using secondary source factors requires a dual-source blending shader.');
            Debug.assert(colorAttachments.length === 1,
                'Dual-source blending requires exactly one color attachment.');
        }
        // each color attachment uses its own blend state - without per-target overrides these all
        // resolve to the state of the target 0
        for (let i = 0; i < colorAttachments.length; i++) {
            const attachmentState = blendState.getAttachment(i, _attachmentBlendState);
            desc.fragment.targets.push({
                format: colorAttachments[i].format,
                writeMask: getWriteMask(attachmentState),
                blend: this.getBlend(attachmentState)
            });
        }

        _pipelineId++;
        DebugHelper.setLabel(desc, `RenderPipelineDescr-${_pipelineId}`);

        return desc;
    }

    /**
     * Creates a render pipeline synchronously. On a cold shader cache this is where the driver
     * compiles the native shader, which can take seconds inside one call.
     *
     * @param {string} primitiveTopology - The primitive topology.
     * @param {number|undefined} ibFormat - The strip index buffer format.
     * @param {Shader} shader - The shader.
     * @param {RenderTarget} renderTarget - The render target.
     * @param {GPUPipelineLayout|string} pipelineLayout - The pipeline layout.
     * @param {BlendState} blendState - The blend state.
     * @param {DepthState} depthState - The depth state.
     * @param {GPUVertexBufferLayout[]} vertexBufferLayout - The vertex buffer layouts.
     * @param {number} cullMode - The cull mode.
     * @param {boolean} stencilEnabled - Whether stencil is enabled.
     * @param {StencilParameters} stencilFront - The stencil state for front faces.
     * @param {StencilParameters} stencilBack - The stencil state for back faces.
     * @param {number} frontFace - The front face.
     * @param {boolean} alphaToCoverageEnabled - Whether alpha to coverage is enabled.
     * @returns {GPURenderPipeline} The pipeline.
     * @private
     */
    create(primitiveTopology, ibFormat, shader, renderTarget, pipelineLayout, blendState, depthState, vertexBufferLayout,
        cullMode, stencilEnabled, stencilFront, stencilBack, frontFace, alphaToCoverageEnabled) {

        const wgpu = this.device.wgpu;

        WebgpuDebug.validate(this.device);

        const desc = this.buildDescriptor(primitiveTopology, ibFormat, shader, renderTarget, pipelineLayout, blendState,
            depthState, vertexBufferLayout, cullMode, stencilEnabled, stencilFront, stencilBack, frontFace,
            alphaToCoverageEnabled);

        this.created.sync++;
        const pipeline = wgpu.createRenderPipeline(desc);

        DebugHelper.setLabel(pipeline, `RenderPipeline-${_pipelineId}`);
        Debug.trace(TRACEID_RENDERPIPELINE_ALLOC, `Alloc: Id ${_pipelineId}, stack: ${DebugGraphics.toString()}`, desc);

        WebgpuDebug.end(this.device, 'RenderPipeline creation', {
            renderPipeline: this,
            desc: desc,
            shader
        });

        return pipeline;
    }

    /**
     * Starts the creation of a render pipeline with createRenderPipelineAsync, which compiles off
     * the calling thread, and stores it in the entry when it resolves.
     *
     * @param {CacheEntry} entry - The cache entry to receive the pipeline.
     * @param {string} primitiveTopology - The primitive topology.
     * @param {number|undefined} ibFormat - The strip index buffer format.
     * @param {Shader} shader - The shader.
     * @param {RenderTarget} renderTarget - The render target.
     * @param {GPUPipelineLayout|string} pipelineLayout - The pipeline layout.
     * @param {BlendState} blendState - The blend state.
     * @param {DepthState} depthState - The depth state.
     * @param {GPUVertexBufferLayout[]} vertexBufferLayout - The vertex buffer layouts.
     * @param {number} cullMode - The cull mode.
     * @param {boolean} stencilEnabled - Whether stencil is enabled.
     * @param {StencilParameters} stencilFront - The stencil state for front faces.
     * @param {StencilParameters} stencilBack - The stencil state for back faces.
     * @param {number} frontFace - The front face.
     * @param {boolean} alphaToCoverageEnabled - Whether alpha to coverage is enabled.
     * @private
     */
    createAsync(entry, primitiveTopology, ibFormat, shader, renderTarget, pipelineLayout, blendState, depthState,
        vertexBufferLayout, cullMode, stencilEnabled, stencilFront, stencilBack, frontFace, alphaToCoverageEnabled) {

        const desc = this.buildDescriptor(primitiveTopology, ibFormat, shader, renderTarget, pipelineLayout, blendState,
            depthState, vertexBufferLayout, cullMode, stencilEnabled, stencilFront, stencilBack, frontFace,
            alphaToCoverageEnabled);

        const id = _pipelineId;
        const generation = this._generation;
        entry.compiling = true;
        this.pending++;
        this.created.async++;

        // After a device loss (clearCache) the entry is unreachable and the compile no longer counts.
        this.device.wgpu.createRenderPipelineAsync(desc).then((pipeline) => {

            // the entry was created synchronously in the meantime, that one wins
            if (!entry.pipeline) {
                DebugHelper.setLabel(pipeline, `RenderPipeline-${id}`);
                Debug.trace(TRACEID_RENDERPIPELINE_ALLOC, `Alloc (async): Id ${id}`, desc);
                entry.pipeline = pipeline;
            }
            this._settled(entry, generation);
        }, (error) => {
            entry.failed = true;
            Debug.warn('Asynchronous render pipeline creation failed, it is created synchronously instead.', error, desc);
            this._settled(entry, generation);
        });
    }
}

export { WebgpuRenderPipeline };
