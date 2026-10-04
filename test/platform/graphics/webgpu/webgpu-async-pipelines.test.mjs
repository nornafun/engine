import { expect } from 'chai';
import sinon from 'sinon';

import { BlendState } from '../../../../src/platform/graphics/blend-state.js';
import { CULLFACE_BACK, CULLFACE_NONE, FRONTFACE_CCW, PRIMITIVE_TRIANGLES } from '../../../../src/platform/graphics/constants.js';
import { DepthState } from '../../../../src/platform/graphics/depth-state.js';
import { StencilParameters } from '../../../../src/platform/graphics/stencil-parameters.js';
import { WebgpuGraphicsDevice } from '../../../../src/platform/graphics/webgpu/webgpu-graphics-device.js';
import { WebgpuMipmapRenderer } from '../../../../src/platform/graphics/webgpu/webgpu-mipmap-renderer.js';
import { WebgpuRenderPipeline } from '../../../../src/platform/graphics/webgpu/webgpu-render-pipeline.js';

const wgsl = `
@vertex fn vmain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(f32(i), 0.0, 0.0, 1.0);
}
@fragment fn fmain() -> @location(0) vec4f {
    return vec4f(1.0);
}`;

// The async path against a real WebGPU implementation (Dawn, null backend: full validation and
// shader compilation, no GPU work), on its own device so that it runs under both `npm test` and
// `npm run test:webgpu`. The engine pieces it does not exercise are replaced by the minimum the
// render pipeline reads.
describe('WebGPU asynchronous pipelines (Dawn)', function () {
    let wgpu;

    before(async function () {
        this.timeout(30000);
        try {
            const { create, globals } = await import('webgpu');
            for (const name of Object.keys(globals)) {
                globalThis[name] ??= globals[name];
            }
            const adapter = await create(['backend=null']).requestAdapter();
            wgpu = await adapter.requestDevice();
        } catch (e) {
            this.skip();
        }
    });

    afterEach(function () {
        sinon.restore();
    });

    describe('render pipelines', function () {
        const stencil = new StencilParameters();
        let device;
        let renderPipeline;
        let shader;

        const renderTarget = {
            samples: 1,
            depth: false,
            stencil: false,
            impl: { key: 1, colorAttachments: [{ format: 'rgba8unorm' }] }
        };

        const get = cullMode => renderPipeline.get({ type: PRIMITIVE_TRIANGLES }, undefined, undefined, undefined,
            shader, renderTarget, [{ key: 1 }, { key: 2 }, { key: 3 }, { key: 4 }], BlendState.NOBLEND,
            DepthState.NODEPTH, cullMode, false, stencil, stencil, FRONTFACE_CCW, false);

        beforeEach(function () {
            const module = wgpu.createShaderModule({ code: wgsl });
            shader = {
                id: 1,
                impl: {
                    getVertexShaderModule: () => module,
                    getFragmentShaderModule: () => module,
                    vertexEntryPoint: 'vmain',
                    fragmentEntryPoint: 'fmain'
                }
            };
            device = { wgpu, asyncPipelines: true };
            renderPipeline = new WebgpuRenderPipeline(device);
            sinon.stub(renderPipeline, 'getPipelineLayout').returns('auto');
            sinon.stub(renderPipeline.vertexBufferLayout, 'get').returns([]);
        });

        it('compiles with createRenderPipelineAsync, pending until it settles', async function () {
            const asyncCreate = sinon.spy(wgpu, 'createRenderPipelineAsync');
            const syncCreate = sinon.spy(wgpu, 'createRenderPipeline');

            expect(get(CULLFACE_BACK)).to.equal(null);
            expect(renderPipeline.pending).to.equal(1);

            await renderPipeline.whenIdle();
            const pipeline = get(CULLFACE_BACK);
            expect(pipeline).to.be.an.instanceof(GPURenderPipeline);
            expect(renderPipeline.pending).to.equal(0);
            expect(asyncCreate.calledOnce).to.be.true;
            expect(syncCreate.called).to.be.false;
            expect(renderPipeline.created).to.deep.equal({ sync: 0, async: 1 });

            // a hit
            expect(get(CULLFACE_BACK)).to.equal(pipeline);
            expect(asyncCreate.calledOnce).to.be.true;
        });

        it('creates the pipeline synchronously with the flag off', function () {
            device.asyncPipelines = false;
            const pipeline = get(CULLFACE_NONE);
            expect(pipeline).to.be.an.instanceof(GPURenderPipeline);
            expect(renderPipeline.created).to.deep.equal({ sync: 1, async: 0 });
        });

        it('takes over a pending pipeline synchronously when the flag is turned off', async function () {
            expect(get(CULLFACE_BACK)).to.equal(null);
            device.asyncPipelines = false;
            const pipeline = get(CULLFACE_BACK);
            expect(pipeline).to.be.an.instanceof(GPURenderPipeline);
            expect(renderPipeline.created).to.deep.equal({ sync: 1, async: 1 });

            await renderPipeline.whenIdle();
            expect(get(CULLFACE_BACK)).to.equal(pipeline);
        });
    });

    describe('mipmap pipelines', function () {
        let renderer;

        beforeEach(function () {
            // the renderer builds its shader from the device, a stand-in with the module is enough
            const module = wgpu.createShaderModule({ code: wgsl });
            const impl = {
                getVertexShaderModule: () => module,
                getFragmentShaderModule: () => module,
                vertexEntryPoint: 'vmain',
                fragmentEntryPoint: 'fmain'
            };
            renderer = Object.create(WebgpuMipmapRenderer.prototype);
            Object.assign(renderer, {
                device: { wgpu },
                shader: { impl },
                pipelineCache: new Map(),
                compiling: new Set(),
                syncCreated: 0
            });
        });

        it('precompiles formats off thread and skips the ones it has', async function () {
            const asyncCreate = sinon.spy(wgpu, 'createRenderPipelineAsync');

            const promise = renderer.precompile(['rgba8unorm', 'rgba16float', 'rgba8unorm']);
            expect(renderer.pending).to.equal(2);
            await promise;

            expect(renderer.pending).to.equal(0);
            expect(renderer.pipelineCache.size).to.equal(2);
            expect(renderer.pipelineCache.get('rgba16float')).to.be.an.instanceof(GPURenderPipeline);
            expect(asyncCreate.callCount).to.equal(2);

            await renderer.precompile(['rgba8unorm', 'rgba16float']);
            expect(asyncCreate.callCount).to.equal(2);
            expect(renderer.syncCreated).to.equal(0);
        });

        it('keeps a pipeline generate created meanwhile, and resolves without error', async function () {
            const promise = renderer.precompile(['rgba8unorm']);
            const sync = {};
            renderer.pipelineCache.set('rgba8unorm', sync);
            await promise;
            expect(renderer.pipelineCache.get('rgba8unorm')).to.equal(sync);
        });

        it('resolves when a compile fails', async function () {
            sinon.stub(wgpu, 'createRenderPipelineAsync').rejects(new Error('failed'));
            const warn = sinon.stub(console, 'warn');
            await renderer.precompile(['rgba8unorm']);
            warn.restore();
            expect(renderer.pending).to.equal(0);
            expect(renderer.pipelineCache.size).to.equal(0);
        });
    });

    describe('device', function () {
        it('adds the pending pipelines of the render and mipmap pipelines', async function () {
            const device = Object.create(WebgpuGraphicsDevice.prototype);
            device.renderPipeline = { pending: 2, created: { sync: 3, async: 4 }, whenIdle: () => Promise.resolve('idle') };
            device.mipmapRenderer = { pending: 1, syncCreated: 5 };

            expect(device.pendingPipelines).to.equal(3);
            expect(device.pipelineCreates).to.deep.equal({ renderSync: 3, renderAsync: 4, mipmapSync: 5 });
            expect(await device.whenPipelinesIdle()).to.equal('idle');
        });
    });
});
