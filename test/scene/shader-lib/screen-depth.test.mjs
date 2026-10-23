import { expect } from 'chai';

import { Preprocessor } from '../../../src/core/preprocessor.js';
import screenDepthGLSL from '../../../src/scene/shader-lib/glsl/chunks/common/frag/screenDepth.js';
import screenDepthWGSL from '../../../src/scene/shader-lib/wgsl/chunks/common/frag/screenDepth.js';

// The depth under the current pixel is read from the scene depth map, a copy of the render target
// the fragment is rendered to. The fragment position is mapped to it by the size of the depth map
// itself, not by screen_size, which is the size of the canvas: a target scaled by
// CameraFrame#rendering.renderTargetScale, or any other offscreen target, has a different size.
describe('Screen depth fragment chunk', function () {

    const chunks = [
        ['GLSL', screenDepthGLSL, 'float getLinearScreenDepth() {', 'gl_FragCoord.xy / vec2(textureSize(uSceneDepthMap, 0))'],
        ['WGSL', screenDepthWGSL, 'fn getLinearScreenDepthFrag() -> f32 {', 'pcPosition.xy / vec2f(textureDimensions(uSceneDepthMap, 0))']
    ];

    for (const [language, chunk, signature, mapping] of chunks) {

        // the body of the function reading the depth under the current pixel
        const pixelDepthBody = () => {
            const source = Preprocessor.run(chunk);
            const start = source.indexOf(signature);
            expect(start).to.be.at.least(0);
            return source.substring(start, source.indexOf('}', start));
        };

        it(`maps the current pixel by the size of the depth map, in ${language}`, function () {
            expect(pixelDepthBody()).to.contain(mapping);
        });

        it(`does not map the current pixel by the size of the canvas, in ${language}`, function () {
            expect(pixelDepthBody()).not.to.contain('screen_size');
        });

        it(`still declares screen_size for the shaders including the chunk, in ${language}`, function () {
            expect(Preprocessor.run(chunk)).to.contain('screen_size');
        });
    }
});
