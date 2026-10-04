import { expect } from 'chai';
import sinon from 'sinon';

import { BlendState } from '../../../../src/platform/graphics/blend-state.js';
import {
    CULLFACE_BACK, CULLFACE_NONE, FRONTFACE_CCW, FUNC_GREATER, FUNC_LESSEQUAL, PRIMITIVE_LINES, PRIMITIVE_POINTS,
    PRIMITIVE_TRIANGLES
} from '../../../../src/platform/graphics/constants.js';
import { DepthState } from '../../../../src/platform/graphics/depth-state.js';
import { StencilParameters } from '../../../../src/platform/graphics/stencil-parameters.js';
import { WebgpuRenderPipeline } from '../../../../src/platform/graphics/webgpu/webgpu-render-pipeline.js';

const createDepthState = (depthBias, depthBiasSlope = 0) => {
    const depthState = new DepthState();
    depthState.depthBias = depthBias;
    depthState.depthBiasSlope = depthBiasSlope;
    return depthState;
};

describe('WebgpuRenderPipeline', function () {

    const shader = { id: 1 };
    const renderTarget = {
        samples: 1,
        depth: true,
        stencil: false,
        impl: { key: 1, depthAttachment: { format: 'depth24plus' } }
    };
    const bindGroupFormats = [{ key: 1 }, { key: 2 }, { key: 3 }, { key: 4 }];
    const stencil = new StencilParameters();

    describe('#get', function () {

        let renderPipeline;

        // looks up the pipeline of a draw differing only in the depth state and the primitive type
        const get = (depthState, type = PRIMITIVE_TRIANGLES) => renderPipeline.get({ type }, undefined, undefined,
            undefined, shader, renderTarget, bindGroupFormats, BlendState.NOBLEND, depthState, CULLFACE_BACK,
            false, stencil, stencil, FRONTFACE_CCW, false);

        beforeEach(function () {
            renderPipeline = new WebgpuRenderPipeline({});
            sinon.stub(renderPipeline, 'getPipelineLayout');
            sinon.stub(renderPipeline.vertexBufferLayout, 'get');
            sinon.stub(renderPipeline, 'create').callsFake(() => ({}));
        });

        afterEach(function () {
            sinon.restore();
        });

        it('shares a pipeline between depth biases truncated to the same integer', function () {
            expect(get(createDepthState(0.5))).to.equal(get(createDepthState(0)));
            expect(get(createDepthState(-1.7))).to.equal(get(createDepthState(-1)));
            expect(get(createDepthState(-2))).to.not.equal(get(createDepthState(-1)));
        });

        it('creates a separate pipeline for a different slope depth bias', function () {
            expect(get(createDepthState(0, 0.5))).to.not.equal(get(createDepthState(0, 0.25)));
            expect(get(createDepthState(0, 0.5))).to.equal(get(createDepthState(0, 0.5)));
        });

        it('ignores the depth bias of points and lines', function () {
            for (const type of [PRIMITIVE_POINTS, PRIMITIVE_LINES]) {
                expect(get(createDepthState(4, 2), type)).to.equal(get(createDepthState(0), type));
            }
            expect(get(createDepthState(4, 2))).to.not.equal(get(createDepthState(0)));
        });

        it('creates a separate pipeline for a different depth function or write', function () {
            const depthStates = [
                DepthState.DEFAULT,
                DepthState.NODEPTH,
                DepthState.WRITEDEPTH,
                new DepthState(FUNC_GREATER),
                new DepthState(FUNC_LESSEQUAL, false)
            ];
            const pipelines = new Set(depthStates.map(depthState => get(depthState)));
            expect(pipelines.size).to.equal(depthStates.length);
        });

    });

    describe('#get with asyncPipelines', function () {

        let device;
        let renderPipeline;
        let compiles;

        const get = (cullMode = CULLFACE_BACK) => renderPipeline.get({ type: PRIMITIVE_TRIANGLES }, undefined, undefined,
            undefined, shader, renderTarget, bindGroupFormats, BlendState.NOBLEND, DepthState.DEFAULT, cullMode,
            false, stencil, stencil, FRONTFACE_CCW, false);

        // a createRenderPipelineAsync whose promises the test settles by hand
        const deferred = () => {
            const d = {};
            d.promise = new Promise((resolve, reject) => {
                d.resolve = resolve;
                d.reject = reject;
            });
            return d;
        };

        beforeEach(function () {
            compiles = [];
            device = {
                asyncPipelines: true,
                wgpu: {
                    createRenderPipelineAsync: sinon.stub().callsFake(() => {
                        const d = deferred();
                        compiles.push(d);
                        return d.promise;
                    })
                }
            };
            renderPipeline = new WebgpuRenderPipeline(device);
            sinon.stub(renderPipeline, 'getPipelineLayout');
            sinon.stub(renderPipeline.vertexBufferLayout, 'get');
            sinon.stub(renderPipeline, 'buildDescriptor').returns({});
            sinon.stub(renderPipeline, 'create').callsFake(() => ({ sync: true }));
        });

        afterEach(function () {
            sinon.restore();
        });

        it('returns null while the pipeline compiles, and compiles each key once', function () {
            expect(get()).to.equal(null);
            expect(get()).to.equal(null);
            expect(renderPipeline.pending).to.equal(1);
            expect(device.wgpu.createRenderPipelineAsync.callCount).to.equal(1);
            expect(renderPipeline.create.called).to.be.false;

            // another key is another compile
            expect(get(CULLFACE_NONE)).to.equal(null);
            expect(renderPipeline.pending).to.equal(2);
        });

        it('returns the pipeline once the compile resolves', async function () {
            expect(get()).to.equal(null);

            const pipeline = {};
            compiles[0].resolve(pipeline);
            await renderPipeline.whenIdle();

            expect(get()).to.equal(pipeline);
            expect(renderPipeline.pending).to.equal(0);
            expect(renderPipeline.created).to.deep.equal({ sync: 0, async: 1 });
        });

        it('settles whenIdle only when every compile has', async function () {
            get();
            get(CULLFACE_NONE);
            let idle = false;
            renderPipeline.whenIdle().then(() => (idle = true));

            compiles[0].resolve({});
            await Promise.resolve();
            await Promise.resolve();
            expect(idle).to.be.false;

            compiles[1].resolve({});
            await renderPipeline.whenIdle();
            expect(idle).to.be.true;
            await renderPipeline.whenIdle();
        });

        it('creates a pipeline synchronously once asyncPipelines is off, and keeps it over the late compile', async function () {
            expect(get()).to.equal(null);

            device.asyncPipelines = false;
            const pipeline = get();
            expect(pipeline).to.deep.equal({ sync: true });

            compiles[0].resolve({ late: true });
            await renderPipeline.whenIdle();
            expect(get()).to.equal(pipeline);
            expect(renderPipeline.pending).to.equal(0);
        });

        it('creates a pipeline synchronously after its compile failed', async function () {
            const warn = sinon.stub(console, 'warn');
            expect(get()).to.equal(null);

            compiles[0].reject(new Error('compile failed'));
            await renderPipeline.whenIdle();
            expect(renderPipeline.pending).to.equal(0);

            expect(get()).to.deep.equal({ sync: true });
            expect(device.wgpu.createRenderPipelineAsync.callCount).to.equal(1);
            warn.restore();
        });

        it('stops counting the compiles of a lost device', async function () {
            sinon.stub(console, 'warn');
            get();
            get(CULLFACE_NONE);
            renderPipeline.clearCache();
            expect(renderPipeline.pending).to.equal(0);
            expect(renderPipeline.cache.size).to.equal(0);

            // one pending compile of the new device
            expect(get()).to.equal(null);
            expect(renderPipeline.pending).to.equal(1);

            compiles[0].resolve({});
            compiles[1].reject(new Error('device lost'));
            await Promise.resolve();
            await Promise.resolve();
            expect(renderPipeline.pending).to.equal(1);

            compiles[2].resolve({});
            await renderPipeline.whenIdle();
            expect(renderPipeline.pending).to.equal(0);
        });

        it('creates synchronously without the flag, or without createRenderPipelineAsync', function () {
            device.asyncPipelines = false;
            expect(get()).to.deep.equal({ sync: true });

            device.asyncPipelines = true;
            delete device.wgpu.createRenderPipelineAsync;
            expect(get(CULLFACE_NONE)).to.deep.equal({ sync: true });
            expect(renderPipeline.pending).to.equal(0);
        });

    });

    describe('#getDepthStencil', function () {

        const getDepthStencil = (depthState, topology) => new WebgpuRenderPipeline({}).getDepthStencil(
            depthState, renderTarget, false, stencil, stencil, topology);

        it('truncates the constant depth bias to an integer', function () {
            const depthStencil = getDepthStencil(createDepthState(-1.7, 0.5), 'triangle-list');
            expect(depthStencil.depthBias).to.equal(-1);
            expect(depthStencil.depthBiasSlopeScale).to.equal(0.5);
        });

        it('does not apply a depth bias to lines', function () {
            const depthStencil = getDepthStencil(createDepthState(-2, 0.5), 'line-list');
            expect(depthStencil.depthBias).to.equal(0);
            expect(depthStencil.depthBiasSlopeScale).to.equal(0);
        });

    });

});
