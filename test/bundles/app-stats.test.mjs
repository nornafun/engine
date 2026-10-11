import { readFileSync } from 'node:fs';
import vm from 'node:vm';

import { expect } from 'chai';
import { JSDOM } from 'jsdom';

import { UMD_TARGETS, createAppFrom } from './helpers.mjs';

describe('build / AppStats', function () {
    for (const target of UMD_TARGETS) {
        it(`reports the plain frame counters in ${target.name}`, function () {
            const dom = new JSDOM('<!doctype html><body></body>', {
                url: 'http://localhost:3210',
                runScripts: 'outside-only'
            });
            let app;
            try {
                vm.runInContext(readFileSync(target.path, 'utf8'), dom.getInternalVMContext());
                app = createAppFrom(dom.window.pc, dom.window.document);

                // incremented by the culler, the renderer and the cluster allocator in every build,
                // and published by updateBasic, which unlike updateDetailed runs in every build
                app.renderer._distanceCulled = 7;
                app.renderer._storageSlotsCopied = 5;
                app.renderer.worldClustersAllocator.uploads = 3;
                app.stats.updateBasic(1000, 0.016, 16, app.renderer, app.graphicsDevice);

                const frame = app.stats.frame;
                expect(frame.distanceCulled).to.equal(7);
                expect(frame.storageSlotCopies).to.equal(5);
                expect(frame.clusterUploads).to.equal(3);
                expect(frame.lightClusters).to.equal(0);

                // reset for the next frame
                app.stats.updateBasic(1016, 0.016, 16, app.renderer, app.graphicsDevice);
                expect(frame.distanceCulled).to.equal(0);
                expect(frame.storageSlotCopies).to.equal(0);
                expect(frame.clusterUploads).to.equal(0);
            } finally {
                app?.destroy();
                dom.window.close();
            }
        });
    }

    for (const target of UMD_TARGETS) {
        it(`measures CPU phases in ${target.name}`, function () {
            const dom = new JSDOM('<!doctype html><body></body>', {
                url: 'http://localhost:3210',
                runScripts: 'outside-only'
            });
            let app;
            try {
                // Install a deterministic clock before the bundle captures performance.now.
                let timestamp = 10000;
                dom.window.performance.now = () => timestamp;
                vm.runInContext(readFileSync(target.path, 'utf8'), dom.getInternalVMContext());
                const pc = dom.window.pc;
                app = createAppFrom(pc, dom.window.document);
                expect(app.stats).to.be.instanceOf(pc.AppStats);

                app.graphicsDevice._primitiveCount = 42;
                app.stats.updateDetailed(app.renderer, app.graphicsDevice);
                const profiled = target.name.startsWith('dbg') || target.name.startsWith('prf');
                expect(app.stats.primitiveCount).to.equal(profiled ? 42 : undefined);
                expect(app.graphicsDevice._primitiveCount).to.equal(profiled ? 0 : 42);

                app.graphicsDevice.update = () => {
                    timestamp += 100;
                };
                app.systems.on('update', () => {
                    timestamp += 3;
                });
                app.systems.on('animationUpdate', () => {
                    timestamp += 2;
                });
                app.systems.on('postUpdate', () => {
                    timestamp += 1;
                });
                app.on('update', () => {
                    timestamp += 4;
                });
                app.inputUpdate = () => {
                    timestamp += 1;
                };
                app.update(0.016);
                expect(app.stats.cpuUpdateTime).to.equal(11);
                expect(app.stats.cpuSystemUpdateTime).to.equal(3);
                expect(app.stats.cpuAnimationTime).to.equal(2);
                expect(app.stats.cpuSystemPostUpdateTime).to.equal(1);

                app.updateCanvasSize = () => {
                    timestamp += 100;
                };
                app.graphicsDevice.frameStart = () => {
                    timestamp += 100;
                };
                app.graphicsDevice.frameEnd = () => {
                    timestamp += 100;
                };
                app.on('prerender', () => {
                    timestamp += 1;
                });
                app.renderComposition = () => {
                    timestamp += 6;
                };
                app.on('postrender', () => {
                    timestamp += 2;
                });
                app.render();
                expect(app.stats.cpuRenderTime).to.equal(9);
                expect(app.stats.gpuFrameTime).to.equal(undefined);
            } finally {
                app?.destroy();
                dom.window.close();
            }
        });
    }
});
