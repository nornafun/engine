import { DebugHelper } from '../../../core/debug.js';
import { DynamicBuffer } from '../dynamic-buffer.js';

class WebgpuDynamicBuffer extends DynamicBuffer {
    /**
     * @type {GPUBuffer}
     * @private
     */
    buffer = null;

    /**
     * CPU access over the whole buffer.
     *
     * @type {ArrayBuffer}
     */
    mappedRange = null;

    /**
     * True for a staging buffer created under {@link WebgpuGraphicsDevice#dynamicBuffersWriteBuffer}:
     * a plain CPU-side ArrayBuffer with no GPUBuffer of its own, written to the gpu buffer with
     * `queue.writeBuffer` instead of a mapped copy. Always false for a gpu buffer.
     *
     * @type {boolean}
     * @private
     */
    cpu = false;

    /**
     * Byte size, tracked directly since a cpu staging buffer has no `buffer.size` to read.
     *
     * @type {number}
     * @private
     */
    size = 0;

    constructor(device, size, isStaging) {
        super(device);

        this.size = size;
        this.cpu = isStaging && device.dynamicBuffersWriteBuffer;

        if (this.cpu) {

            // a plain CPU buffer: no GPU resource, no mapAsync round trip, written to the gpu
            // buffer with queue.writeBuffer at submit (webgpu-dynamic-buffers.js#submit)
            this.mappedRange = new ArrayBuffer(size);
            this.setStorage(this.mappedRange);
        } else {

            this.buffer = device.wgpu.createBuffer({
                size: size,
                usage: isStaging ? GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC : GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                mappedAtCreation: isStaging
            });

            if (isStaging) {
                this.onAvailable();
            }

            DebugHelper.setLabel(this.buffer, `DynamicBuffer-${isStaging ? 'Staging' : 'Gpu'}`);
        }

        // staging buffers are not stored in vram, but add them for tracking purposes anyway
        device._vram.ub += size;
    }

    destroy(device) {

        this.bindGroupCache.forEach(bindGroup => bindGroup.destroy());
        this.bindGroupCache.clear();
        this.bindGroupFormat.destroy();
        this.mappedRange = null;
        this.setStorage(null);

        device._vram.ub -= this.size;

        if (this.buffer) {
            this.buffer.destroy();
            this.buffer = null;
        }
    }

    /**
     * Called when the staging buffer is mapped for writing. Never called for a cpu staging buffer:
     * its storage is set once, in the constructor, since it is never unmapped.
     */
    onAvailable() {
        // map the whole buffer - each mapping returns new memory, so the storage views are created
        // for it here, once per mapping instead of once per allocation
        this.mappedRange = this.buffer.getMappedRange();
        this.setStorage(this.mappedRange);
    }
}

export { WebgpuDynamicBuffer };
