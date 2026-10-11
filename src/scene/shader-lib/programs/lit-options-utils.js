import { LIGHTTYPE_DIRECTIONAL } from '../../constants.js';

const LitOptionsUtils = {

    // generate a key for the lit options
    generateKey(options) {
        return `lit${Object.keys(options)
        .sort()
        .map((key) => {
            if (key === 'shaderChunks') {
                return options.shaderChunks?.key ?? '';
            } else if (key === 'lights') {
                return LitOptionsUtils.generateLightsKey(options);
            } else if (key === 'noShadow') {
                // the flag only removes the shadows of lights casting them, so without such a light
                // the shader does not depend on it, and is keyed as if it was not set
                return key + (options.noShadow && LitOptionsUtils.readsShadows(options));
            }
            return key + options[key];
        })
        .join('\n')}`;
    },

    /**
     * Returns true when the shadows the lit shader receives depend on the options, which are then
     * the only ones the noShadow option removes: the clustered shadows, or the shadows of a light
     * which casts them and which the shader evaluates itself - every light without clustered
     * lighting, and the directional lights with it. Mirrors LitShader#_setupLightingDefines.
     *
     * @param {object} options - The lit shader options.
     * @returns {boolean} True when the shader reads the noShadow option.
     */
    readsShadows(options) {
        const clustered = options.clusteredLightingEnabled;
        if (clustered && options.clusteredLightingShadowsEnabled) {
            return true;
        }
        const lights = options.lights;
        for (let i = 0; i < lights.length; i++) {
            const light = lights[i];
            if (light && light.castShadows && (!clustered || light._type === LIGHTTYPE_DIRECTIONAL)) {
                return true;
            }
        }
        return false;
    },

    generateLightsKey(options) {
        // A light's slot is part of the shader it generates - light<N>_ names are emitted per slot -
        // and `options.lights` is indexed by slot and sparse, so the slot each light sits in has to
        // be in the key. Keying on the sequence of lights alone would let two layouts of the same
        // lights, say slots [0, 1] and [0, 2], share a cached shader declaring the wrong slot.
        const lights = options.lights;
        let key = 'lights:';
        for (let i = 0; i < lights.length; i++) {
            const light = lights[i];
            if (light && (!options.clusteredLightingEnabled || light._type === LIGHTTYPE_DIRECTIONAL)) {
                key += `${i}:${light.key},`;
            }
        }
        return key;
    }
};

export { LitOptionsUtils };
