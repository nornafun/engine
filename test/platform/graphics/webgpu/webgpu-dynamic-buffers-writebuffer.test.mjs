import { expect } from 'chai';
import sinon from 'sinon';

import { WebgpuDynamicBuffer } from '../../../../src/platform/graphics/webgpu/webgpu-dynamic-buffer.js';
import { WebgpuDynamicBuffers } from '../../../../src/platform/graphics/webgpu/webgpu-dynamic-buffers.js';
import { WebgpuGraphicsDevice } from '../../../../src/platform/graphics/webgpu/webgpu-graphics-device.js';

// WebgpuGraphicsDevice#dynamicBuffersWriteBuffer: queue.writeBuffer instead of a staging ring and
// mapBufferAsync. Against a real WebGPU implementation (Dawn, null backend) like
// webgpu-async-pipelines.test.mjs, so GPUBuffer/ArrayBuffer identity checks are real, not mocked.
describe('WebGPU dynamic buffers writeBuffer path (Dawn)', function () {
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

    // real prototype (so BindGroupFormat's createBindGroupFormatImpl etc. work), minimum state set
    const fakeDevice = writeBuffer => Object.assign(Object.create(WebgpuGraphicsDevice.prototype), {
        wgpu,
        dynamicBuffersWriteBuffer: writeBuffer,
        _vram: { ub: 0 },
        _bindGroupFormats: new Set(),
        addCommandBuffer: sinon.stub()
    });

    describe('WebgpuDynamicBuffer', function () {
        it('is a plain CPU ArrayBuffer, no GPUBuffer, when the flag is on', function () {
            const device = fakeDevice(true);
            const buf = new WebgpuDynamicBuffer(device, 256, true);
            expect(buf.cpu).to.be.true;
            expect(buf.buffer).to.equal(null);
            expect(buf.mappedRange).to.be.an.instanceof(ArrayBuffer);
            expect(buf.mappedRange.byteLength).to.equal(256);
            expect(device._vram.ub).to.equal(256);

            buf.destroy(device);
            expect(device._vram.ub).to.equal(0);
        });

        it('stays a real mapped GPUBuffer when the flag is off, same as before', function () {
            const device = fakeDevice(false);
            const buf = new WebgpuDynamicBuffer(device, 256, true);
            expect(buf.cpu).to.be.false;
            expect(buf.buffer).to.be.an.instanceof(GPUBuffer);
            // Dawn's getMappedRange() ArrayBuffer crosses a realm boundary under mocha, so byteLength
            // (not instanceof, which fails cross-realm) is the honest check here.
            expect(buf.mappedRange?.byteLength).to.equal(256);

            buf.destroy(device);
            expect(device._vram.ub).to.equal(0);
        });

        it('a non-staging (gpu) buffer is never cpu, whatever the flag', function () {
            const device = fakeDevice(true);
            const buf = new WebgpuDynamicBuffer(device, 256, false);
            expect(buf.cpu).to.be.false;
            expect(buf.buffer).to.be.an.instanceof(GPUBuffer);
        });
    });

    describe('WebgpuDynamicBuffers#submit', function () {
        it('writeBuffer path: queue.writeBuffer once, both buffers reusable immediately, nothing pending', function () {
            const device = fakeDevice(true);
            const writeBufferSpy = sinon.spy(wgpu.queue, 'writeBuffer');
            const dyn = new WebgpuDynamicBuffers(device, 1024, 256);

            const allocation = {};
            dyn.alloc(allocation, 64);
            new Uint8Array(allocation.storageBuffer.mappedRange, 0, 64).fill(7);

            dyn.submit();

            expect(writeBufferSpy.calledOnce).to.be.true;
            const [gpuBuf, bufferOffset, data, dataOffset, size] = writeBufferSpy.firstCall.args;
            expect(gpuBuf).to.be.an.instanceof(GPUBuffer);
            expect(bufferOffset).to.equal(0);
            expect(data).to.be.an.instanceof(ArrayBuffer);
            expect(dataOffset).to.equal(0);
            expect(size).to.equal(64);

            expect(dyn.usedBuffers.length).to.equal(0);
            expect(dyn.pendingStagingBuffers.length).to.equal(0);
            expect(dyn.gpuBuffers.length).to.equal(1);
            expect(dyn.stagingBuffers.length).to.equal(1);
            expect(device.addCommandBuffer.called).to.be.false;

            dyn.destroy();
        });

        it('default path: command-buffer copy, staging buffer pending until onCommandBuffersSubmitted', function () {
            const device = fakeDevice(false);
            const dyn = new WebgpuDynamicBuffers(device, 1024, 256);

            const allocation = {};
            dyn.alloc(allocation, 64);
            dyn.submit();

            expect(device.addCommandBuffer.calledOnce).to.be.true;
            expect(dyn.usedBuffers.length).to.equal(0);
            expect(dyn.pendingStagingBuffers.length).to.equal(1);
            expect(dyn.gpuBuffers.length).to.equal(1);
            expect(dyn.stagingBuffers.length).to.equal(0);

            dyn.destroy();
        });
    });
});
