import { expect } from 'chai';
import sinon from 'sinon';

import { Entity } from '../../../src/framework/entity.js';
import { Texture } from '../../../src/platform/graphics/texture.js';
import { LAYERID_WORLD } from '../../../src/scene/constants.js';
import { Layer } from '../../../src/scene/layer.js';
import { createApp } from '../../app.mjs';
import { jsdomSetup, jsdomTeardown } from '../../jsdom.mjs';

describe('WorldClusters texture uploads', function () {
    let app;
    let lamp;
    let uploads;

    beforeEach(function () {
        jsdomSetup();
        app = createApp();

        const camera = new Entity('camera');
        camera.addComponent('camera');
        camera.setPosition(0, 0, 10);
        app.root.addChild(camera);

        const box = new Entity('box');
        box.addComponent('render', { type: 'box' });
        app.root.addChild(box);

        lamp = new Entity('lamp');
        lamp.addComponent('light', { type: 'omni', range: 5 });
        lamp.setPosition(1, 1, 1);
        app.root.addChild(lamp);

        uploads = { LightsTexture: 0, ClusterTexture: 0 };
        const upload = Texture.prototype.upload;
        sinon.stub(Texture.prototype, 'upload').callsFake(function () {
            if (this.name in uploads) {
                uploads[this.name]++;
            }
            return upload.call(this);
        });
    });

    afterEach(function () {
        sinon.restore();
        app.destroy();
        jsdomTeardown();
    });

    const counts = () => ({ ...uploads });

    it('uploads the lights and the clusters once, then not again while they do not change', function () {
        app.render();
        expect(app.renderer.worldClustersAllocator.count).to.equal(1);
        expect(counts()).to.deep.equal({ LightsTexture: 1, ClusterTexture: 1 });

        app.render();
        app.render();
        expect(counts()).to.deep.equal({ LightsTexture: 1, ClusterTexture: 1 });
    });

    it('uploads the lights again when a light changes', function () {
        app.render();
        const before = counts();

        lamp.light.intensity = 0.5;
        app.render();
        expect(uploads.LightsTexture).to.equal(before.LightsTexture + 1);

        app.render();
        expect(uploads.LightsTexture).to.equal(before.LightsTexture + 1);
    });

    it('uploads the clusters again when a light moves relative to the others', function () {
        const other = new Entity('other');
        other.addComponent('light', { type: 'omni', range: 2 });
        other.setPosition(-3, 1, 1);
        app.root.addChild(other);

        app.render();
        const before = counts();

        other.setPosition(3, 3, 1);
        app.render();
        expect(uploads.LightsTexture).to.equal(before.LightsTexture + 1);
        expect(uploads.ClusterTexture).to.be.above(before.ClusterTexture);
    });

    it('uploads both again when a light is removed', function () {
        app.render();
        const before = counts();

        lamp.enabled = false;
        app.render();
        expect(uploads.LightsTexture).to.be.at.most(before.LightsTexture + 1);
        app.render();
        const after = counts();
        app.render();
        expect(counts()).to.deep.equal(after);
    });

    it('allocates no cluster for a layer with only directional lights, and uploads nothing for it', function () {
        const worldLayer = app.scene.layers.getLayerById(LAYERID_WORLD);
        const layer = new Layer({ name: 'Viewmodel' });
        app.scene.layers.push(layer);

        const sun = new Entity('sun');
        sun.addComponent('light', { type: 'directional', layers: [worldLayer.id, layer.id] });
        app.root.addChild(sun);

        const tool = new Entity('tool');
        tool.addComponent('render', { type: 'box', layers: [layer.id] });
        app.root.addChild(tool);

        const camera = app.root.findByName('camera').camera;
        camera.layers = [worldLayer.id, layer.id];

        app.render();
        expect(layer.hasClusteredLights).to.equal(false);
        expect(layer.meshInstances.length).to.be.above(0);
        expect(app.renderer.worldClustersAllocator.count).to.equal(1);
        expect(counts()).to.deep.equal({ LightsTexture: 1, ClusterTexture: 1 });
    });

    it('counts the uploads in the frame stats', function () {
        const stats = app.stats;
        const frameUploads = () => {
            app.render();
            stats.updateBasic(1000, 0.016, 16, app.renderer, app.graphicsDevice);
            return stats.frame.clusterUploads;
        };

        expect(frameUploads()).to.equal(2);
        expect(frameUploads()).to.equal(0);
        lamp.light.intensity = 0.5;
        expect(frameUploads()).to.equal(1);
        expect(stats.frame.lightClusters).to.equal(1);
    });
});
