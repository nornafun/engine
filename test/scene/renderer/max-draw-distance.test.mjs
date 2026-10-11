import { expect } from 'chai';

import { Entity } from '../../../src/framework/entity.js';
import { LAYERID_WORLD } from '../../../src/scene/constants.js';
import { createApp } from '../../app.mjs';
import { createGraphicsDevice } from '../../device.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

// the half diagonal of a unit box, the radius of its bounding sphere
const RADIUS = Math.sqrt(0.75);

describe('Max draw distance', function () {
    let app;
    let camera;
    let layer;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();

        const entity = new Entity('camera');
        entity.addComponent('camera');
        app.root.addChild(entity);
        camera = entity.camera;
        layer = app.scene.layers.getLayerById(LAYERID_WORLD);
    });

    afterEach(function () {
        app.destroy();
        jsdomTeardown();
    });

    // a unit box straight ahead of the camera, which looks down -Z from the origin
    const addBox = (distance, castShadow = false) => {
        const entity = new Entity(`box${distance}`);
        entity.addComponent('render', { type: 'box', castShadows: castShadow });
        entity.setPosition(0, 0, -distance);
        app.root.addChild(entity);
        return entity.render.meshInstances[0];
    };

    const rendered = () => {
        app.render();
        return layer.getCulledInstances(camera.camera).opaque;
    };

    it('is off by default', function () {
        const far = addBox(900);
        expect(far.maxDrawDistance).to.equal(0);
        expect(layer.maxDrawDistance).to.equal(0);
        expect(rendered()).to.include(far);
        expect(app.renderer._distanceCulled).to.equal(0);
    });

    it('culls a mesh instance when its whole bounding sphere is beyond the distance', function () {
        const inside = addBox(30);
        const edge = addBox(30 + RADIUS - 0.01);
        const outside = addBox(30 + RADIUS + 0.01);
        for (const meshInstance of [inside, edge, outside]) {
            meshInstance.maxDrawDistance = 30;
        }

        const list = rendered();
        expect(list).to.include(inside);
        expect(list).to.include(edge);
        expect(list).not.to.include(outside);
        expect(app.renderer._distanceCulled).to.equal(1);
    });

    it('culls by distance in every direction, behind and beside the camera as well', function () {
        const behind = addBox(-20);
        const to_the_side = addBox(0);
        to_the_side.node.setPosition(40, 0, -1);
        behind.maxDrawDistance = 10;
        to_the_side.maxDrawDistance = 10;
        to_the_side.cull = true;

        // both are also outside the frustum, so the count tells the distance culled them first
        const list = rendered();
        expect(list).not.to.include(behind);
        expect(list).not.to.include(to_the_side);
        expect(app.renderer._distanceCulled).to.equal(2);
    });

    it('inherits the distance of the layer, which a mesh instance overrides', function () {
        const far = addBox(50);
        const farOwn = addBox(50);
        const farInfinite = addBox(50);
        const near = addBox(10);
        farOwn.maxDrawDistance = 100;
        farInfinite.maxDrawDistance = Infinity;
        layer.maxDrawDistance = 25;

        const list = rendered();
        expect(list).not.to.include(far);
        expect(list).to.include(farOwn);
        expect(list).to.include(farInfinite);
        expect(list).to.include(near);
        expect(app.renderer._distanceCulled).to.equal(1);

        layer.maxDrawDistance = 0;
        expect(rendered()).to.include(far);
    });

    it('never culls a mesh instance with culling disabled, or when the camera does not cull', function () {
        const noCull = addBox(500);
        noCull.maxDrawDistance = 10;
        noCull.cull = false;
        expect(rendered()).to.include(noCull);

        noCull.cull = true;
        camera.camera.frustumCulling = false;
        expect(rendered()).to.include(noCull);
        expect(app.renderer._distanceCulled).to.equal(0);
    });

    it('measures from the position of the camera', function () {
        const box = addBox(50);
        box.maxDrawDistance = 20;
        expect(rendered()).not.to.include(box);

        camera.entity.setPosition(0, 0, -35);
        expect(rendered()).to.include(box);
    });

    it('leaves the shader variants and the light hash alone', function () {
        const box = addBox(10);
        app.render();
        const shaderDefs = box._shaderDefs;
        const key = box.key;
        box.maxDrawDistance = 5;
        layer.maxDrawDistance = 5;
        app.render();
        expect(box._shaderDefs).to.equal(shaderDefs);
        expect(box.key).to.equal(key);
    });

    it('reports the count in the frame stats', function () {
        const far = addBox(80);
        far.maxDrawDistance = 10;
        app.render();
        app.stats.frame.distanceCulled = -1;
        app.stats.update?.(0);
        app.stats.updateDetailed(app.renderer, app.graphicsDevice);
        expect(app.stats.frame.distanceCulled).to.equal(1);
        expect(app.renderer._distanceCulled).to.equal(0);
    });

    describe('shadow casters', function () {

        let light;

        beforeEach(function () {
            const entity = new Entity('light');
            entity.addComponent('light', { type: 'directional', castShadows: true, numCascades: 1, shadowDistance: 500 });
            entity.setEulerAngles(60, 30, 0);
            app.root.addChild(entity);
            light = entity.light.light;
        });

        const casters = () => light.getRenderData(camera.camera, 0).visibleCasters;

        it('are culled by their distance from the camera, not from the shadow camera', function () {
            const near = addBox(10, true);
            const far = addBox(60, true);
            const farUnlimited = addBox(60, true);
            near.maxDrawDistance = 30;
            far.maxDrawDistance = 30;

            app.render();
            expect(casters()).to.include(near);
            expect(casters()).to.include(farUnlimited);
            expect(casters()).not.to.include(far);
            expect(app.renderer._distanceCulled).to.be.at.least(1);
        });

        it('use the distance of the layer of the casters', function () {
            const far = addBox(60, true);
            layer.maxDrawDistance = 30;
            app.render();
            expect(casters()).not.to.include(far);

            far.maxDrawDistance = Infinity;
            app.render();
            expect(casters()).to.include(far);
        });
    });
});

describe('Mesh instance storage device option', function () {

    let device;

    afterEach(function () {
        device?.destroy();
        device = null;
    });

    it('is on by default and can be turned off before the first shader', function () {
        jsdomSetup();
        try {
            const canvas = document.createElement('canvas');
            device = createGraphicsDevice(canvas);
            const supported = device.supportsMeshInstanceStorage;
            expect(supported).to.equal(device.isWebGPU);
            expect(!!device.meshInstanceStorage).to.equal(supported);
            device.destroy();

            device = createGraphicsDevice(document.createElement('canvas'), { meshInstanceStorage: false });
            expect(device.supportsMeshInstanceStorage).to.equal(false);
            expect(device.meshInstanceStorage).to.equal(null);
        } finally {
            jsdomTeardown();
        }
    });
});
