import { DebugHelper } from '../../../core/debug.js';
import { DebugGraphics } from '../debug-graphics.js';
import { DynamicBuffers } from '../dynamic-buffers.js';
import { WebgpuDynamicBuffer } from './webgpu-dynamic-buffer.js';

class WebgpuDynamicBuffers extends DynamicBuffers {
    /**
     * Staging buffers which are getting copied over to gpu buffers in the command buffer waiting
     * to be submitted. When those command buffers are submitted, we can mapAsync these staging
     * buffers for reuse.
     *
     * @type {WebgpuDynamicBuffer[]}
     */
    pendingStagingBuffers = [];

    destroy() {
        // Loss can interrupt a frame before its active allocations have been submitted.
        this.scheduleSubmit();
        for (const { gpuBuffer, stagingBuffer } of this.usedBuffers) {
            gpuBuffer.destroy(this.device);
            stagingBuffer.destroy(this.device);
        }
        for (const stagingBuffer of this.pendingStagingBuffers) {
            stagingBuffer.destroy(this.device);
        }
        this.pendingStagingBuffers.length = 0;
        super.destroy();
    }

    createBuffer(device, size, isStaging) {
        return new WebgpuDynamicBuffer(device, size, isStaging);
    }

    /**
     * Submit all used buffers to the device.
     */
    submit() {

        super.submit();

        // submit all used buffers
        const count = this.usedBuffers.length;
        if (count) {

            const device = this.device;

            if (device.dynamicBuffersWriteBuffer) {
                this.submitWriteBuffer(count);
            } else {
                this.submitStaging(count);
            }
        }
    }

    /**
     * The default path: copies each used range from its CPU-mapped staging buffer to its gpu
     * buffer with a command-buffer copy, and queues the staging buffer to be remapped for reuse
     * once that command buffer's submit settles (`onCommandBuffersSubmitted`).
     *
     * @param {number} count - `this.usedBuffers.length`, already known non-zero by `submit`.
     * @private
     */
    submitStaging(count) {

        const device = this.device;
        const gpuBuffers = this.gpuBuffers;

        // new command encoder, as buffer copies need to be submitted before the currently recorded
        // rendering encoder is submitted
        const commandEncoder = device.wgpu.createCommandEncoder();
        DebugHelper.setLabel(commandEncoder, 'DynamicBuffersSubmit');

        DebugGraphics.pushGpuMarker(device, 'DynamicBuffersSubmit');

        // run this loop backwards to preserve the order of buffers in gpuBuffers array
        for (let i = count - 1; i >= 0; i--) {
            const usedBuffer = this.usedBuffers[i];
            const { stagingBuffer, gpuBuffer, offset, size } = usedBuffer;

            // unmap staging buffer (we're done writing to it on CPU)
            const src = stagingBuffer.buffer;
            src.unmap();

            // schedule data copy from staging to gpu buffer
            commandEncoder.copyBufferToBuffer(src, offset, gpuBuffer.buffer, offset, size);

            // gpu buffer can be reused immediately
            gpuBuffers.push(gpuBuffer);
        }

        DebugGraphics.popGpuMarker(device);

        // schedule the command buffer to run before all currently scheduled command buffers
        const cb = commandEncoder.finish();
        DebugHelper.setLabel(cb, 'DynamicBuffers');
        device.addCommandBuffer(cb, true);

        // keep the used staging buffers in the pending list
        for (let i = 0; i < count; i++) {
            const stagingBuffer = this.usedBuffers[i].stagingBuffer;
            this.pendingStagingBuffers.push(stagingBuffer);
        }
        this.usedBuffers.length = 0;
    }

    /**
     * {@link WebgpuGraphicsDevice#dynamicBuffersWriteBuffer} path: each used range is a plain
     * CPU ArrayBuffer (no GPUBuffer, no mapAsync), copied to the gpu buffer with `queue.writeBuffer`
     * - the browser schedules and paces this copy itself, outside any command buffer. Both the gpu
     * buffer and the staging buffer are reusable the moment this returns: `writeBuffer` reads its
     * source synchronously (the spec requires the data be captured before the call returns), so
     * there is no pending list and `onCommandBuffersSubmitted` has nothing to do for these.
     *
     * @param {number} count - `this.usedBuffers.length`, already known non-zero by `submit`.
     * @private
     */
    submitWriteBuffer(count) {

        const device = this.device;
        const queue = device.wgpu.queue;
        const gpuBuffers = this.gpuBuffers;
        const stagingBuffers = this.stagingBuffers;

        DebugGraphics.pushGpuMarker(device, 'DynamicBuffersSubmit-WriteBuffer');

        for (let i = count - 1; i >= 0; i--) {
            const usedBuffer = this.usedBuffers[i];
            const { stagingBuffer, gpuBuffer, offset, size } = usedBuffer;

            queue.writeBuffer(gpuBuffer.buffer, offset, stagingBuffer.mappedRange, offset, size);

            // both buffers are reusable immediately - no staging ring, no mapBufferAsync wait
            gpuBuffers.push(gpuBuffer);
            stagingBuffers.push(stagingBuffer);
        }

        DebugGraphics.popGpuMarker(device);

        this.usedBuffers.length = 0;
    }

    /**
     * Called when all scheduled command buffers are submitted to the device.
     */
    onCommandBuffersSubmitted() {
        // map the staging buffers for write to allow them to be reused - this resolves when the CBs
        // using them are done on the GPU
        const count = this.pendingStagingBuffers.length;
        if (count) {
            const device = this.device;
            for (let i = 0; i < count; i++) {
                const stagingBuffer = this.pendingStagingBuffers[i];
                device.mapBufferAsync(stagingBuffer.buffer, GPUMapMode.WRITE).then((mapped) => {
                    // the mapping fails when the device is lost, and can also complete after the
                    // device has been destroyed - in either case the buffer cannot be reused, so
                    // destroy it to release its memory and vram tracking
                    if (mapped && this.stagingBuffers) {
                        stagingBuffer.onAvailable();
                        this.stagingBuffers.push(stagingBuffer);
                    } else {
                        stagingBuffer.destroy(device);
                    }
                });
            }
            this.pendingStagingBuffers.length = 0;
        }
    }
}

export { WebgpuDynamicBuffers };
