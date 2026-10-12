import { SHADERLANGUAGE_GLSL, SHADERLANGUAGE_WGSL } from '../../platform/graphics/constants.js';
import { DeviceCache } from '../../platform/graphics/device-cache.js';
import { Preprocessor } from '../../core/preprocessor.js';
import { ShaderChunks } from './shader-chunks.js';

/**
 * @import { GraphicsDevice } from '../../platform/graphics/graphics-device.js'
 * @import { ShaderChunkMap } from './shader-chunk-map.js'
 * @import { ShaderPassInfo } from '../shader-pass.js'
 */

// The defines naming the built-in shader passes. A pass defines its own and none of the others, so
// the branches of the lit shaders selected by them are known before a shader is generated.
const PASS_DEFINES = ['FORWARD_PASS', 'PREPASS_PASS', 'SHADOW_PASS', 'PICK_PASS', 'DEPTH_PICK_PASS'];

// The roots of a standard shader: the main chunks of both stages, and the instancing chunk which the
// generator compares to the engine's own. The includes the generator emits itself name the chunks
// they are copied from.
const ROOTS = ['litMainVS', 'litMainPS', 'transformInstancingVS'];
const ALIASES = new Map([
    ['frontendDeclPS', 'stdDeclarationPS'],
    ['frontendCodePS', 'stdFrontEndPS']
]);

const DIRECTIVE = /#(ifdef|ifndef|if|elif|else|endif|include)\b([^\n]*)/g;
const INCLUDE_NAME = /^\s*"([\w-]+)/;
const PASS_DEFINE_WRITE = new RegExp(`#\\s*(?:define|undef)\\s+(?:${PASS_DEFINES.join('|')})\\b`);
const TOKEN = /\s*(\|\||&&|[!()]|defined\b|\w+)/y;

const UNKNOWN = 2;

/**
 * Evaluates a preprocessor condition built of defined(), !, &&, || and parentheses, knowing only
 * which of the pass defines are set. Returns true, false, or UNKNOWN when the condition depends on
 * anything else.
 *
 * @param {string} expression - The condition.
 * @param {Set<string>} defined - The pass defines the pass sets.
 * @returns {boolean|number} The value of the condition.
 */
const evaluate = (expression, defined) => {
    const tokens = [];
    TOKEN.lastIndex = 0;
    const source = expression.trim();
    while (TOKEN.lastIndex < source.length) {
        const match = TOKEN.exec(source);
        if (!match) return UNKNOWN;
        tokens.push(match[1]);
    }

    let i = 0;
    let valid = true;

    const not = v => (v === UNKNOWN ? UNKNOWN : !v);
    const and = (a, b) => (a === false || b === false ? false : (a === UNKNOWN || b === UNKNOWN ? UNKNOWN : true));
    const or = (a, b) => (a === true || b === true ? true : (a === UNKNOWN || b === UNKNOWN ? UNKNOWN : false));

    // the conditions nest, so the parts refer to each other, as hoisted functions
    function unary() {
        const token = tokens[i++];
        if (token === '!') return not(unary());
        if (token === '(') {
            const value = disjunction();
            if (tokens[i++] !== ')') valid = false;
            return value;
        }
        if (token === 'defined') {
            const parens = tokens[i] === '(';
            if (parens) i++;
            const name = tokens[i++];
            if (parens && tokens[i++] !== ')') valid = false;
            if (!name || !/^\w+$/.test(name)) valid = false;
            return PASS_DEFINES.includes(name) ? defined.has(name) : UNKNOWN;
        }
        valid = false;
        return UNKNOWN;
    }

    function conjunction() {
        let value = unary();
        while (tokens[i] === '&&') {
            i++;
            value = and(value, unary());
        }
        return value;
    }

    function disjunction() {
        let value = conjunction();
        while (tokens[i] === '||') {
            i++;
            value = or(value, conjunction());
        }
        return value;
    }

    const value = disjunction();
    return valid && i === tokens.length ? value : UNKNOWN;
};

/**
 * Collects the names of the chunks a shader of a pass can include: every chunk an #include reaches
 * from the roots, through every branch except the ones the pass defines rule out. Returns null when
 * that cannot be told from the text - a chunk writing a pass define, or one whose conditional
 * blocks do not close within it.
 *
 * @param {(name: string) => string|undefined} chunkOf - The code of a chunk.
 * @param {Set<string>} defined - The pass defines the pass sets.
 * @returns {Set<string>|null} The names of the reachable chunks.
 */
const reachableChunks = (chunkOf, defined) => {
    const reached = new Set();
    const pending = ROOTS.slice();

    while (pending.length) {
        const name = pending.pop();
        if (reached.has(name)) continue;
        reached.add(name);

        const code = chunkOf(name);
        if (code === undefined) continue;
        const source = Preprocessor.stripComments(code);
        if (PASS_DEFINE_WRITE.test(source)) return null;

        // each block: whether it can be taken, and whether a branch before it was surely taken
        const stack = [];
        const live = () => stack.length === 0 || stack[stack.length - 1].live;

        DIRECTIVE.lastIndex = 0;
        let match;
        while ((match = DIRECTIVE.exec(source)) !== null) {
            const [, keyword, rest] = match;
            switch (keyword) {
                case 'ifdef':
                case 'ifndef':
                case 'if': {
                    const name = rest.trim();
                    let value = keyword === 'if' ? evaluate(rest, defined) :
                        (PASS_DEFINES.includes(name) ? defined.has(name) : UNKNOWN);
                    if (keyword === 'ifndef' && value !== UNKNOWN) value = !value;
                    const parent = live();
                    stack.push({ parent, taken: value === true, live: parent && value !== false });
                    break;
                }
                case 'elif': {
                    const block = stack[stack.length - 1];
                    if (!block) return null;
                    const value = evaluate(rest, defined);
                    block.live = block.parent && !block.taken && value !== false;
                    block.taken ||= value === true;
                    break;
                }
                case 'else': {
                    const block = stack[stack.length - 1];
                    if (!block) return null;
                    block.live = block.parent && !block.taken;
                    block.taken = true;
                    break;
                }
                case 'endif':
                    if (!stack.pop()) return null;
                    break;
                case 'include': {
                    const include = INCLUDE_NAME.exec(rest);
                    if (include && live()) {
                        pending.push(ALIASES.get(include[1]) ?? include[1]);
                    }
                    break;
                }
            }
        }
        if (stack.length) return null;
    }

    return reached;
};

const _cache = new DeviceCache();

/**
 * Returns the shader chunks of a material which a shader of a non-forward pass can include: the
 * ones the pass reads, without the overrides of the forward code it never includes, so materials
 * differing in those alone share the shader. Generated from these chunks, the shader is the one
 * generated from all of them - the chunks left out are never included, and are only scanned for
 * the vertex attributes they would use.
 *
 * @param {GraphicsDevice} device - The graphics device.
 * @param {ShaderChunks} chunks - The shader chunks of the material.
 * @param {ShaderPassInfo} shaderPassInfo - The pass, which must not be a forward pass.
 * @param {Map<string, string>} [defines] - The defines of the shader, whose pass defines must be the
 * pass's own.
 * @returns {ShaderChunks} The chunks, or the given ones when nothing is left out.
 */
const getPassShaderChunks = (device, chunks, shaderPassInfo, defines) => {
    if (!chunks || (chunks.glsl.size === 0 && chunks.wgsl.size === 0) || shaderPassInfo.isForward) {
        return chunks;
    }

    // the reachable chunks follow from the engine chunks too, which an application can change
    const engineGLSL = ShaderChunks.get(device, SHADERLANGUAGE_GLSL);
    const engineWGSL = ShaderChunks.get(device, SHADERLANGUAGE_WGSL);
    const cache = _cache.get(device, () => ({ glslKey: null, wgslKey: null, results: new Map() }));
    if (cache.glslKey !== engineGLSL.key || cache.wgslKey !== engineWGSL.key) {
        cache.glslKey = engineGLSL.key;
        cache.wgslKey = engineWGSL.key;
        cache.results.clear();
    }

    // the pass defines the shader is generated with must be the pass's own
    const defined = new Set(PASS_DEFINES.filter(name => shaderPassInfo.defines.has(name)));
    if (defines && PASS_DEFINES.some(name => defines.has(name) !== defined.has(name))) {
        return chunks;
    }

    const key = `${[...defined].join(',')}|${chunks.glsl.key}|${chunks.wgsl.key}`;

    // null when nothing is left out: the material's own chunks are used, never another's
    let result = cache.results.get(key);
    if (result === undefined) {

        /**
         * @param {ShaderChunkMap} user - The chunks of the material in one language.
         * @param {ShaderChunkMap} engine - The chunks of the engine in that language.
         * @returns {Set<string>|null} The chunks a shader of the pass can include.
         */
        const reach = (user, engine) => {
            return user.size === 0 ? new Set() : reachableChunks(name => user.get(name) ?? engine.get(name), defined);
        };

        const glsl = reach(chunks.glsl, engineGLSL);
        const wgsl = reach(chunks.wgsl, engineWGSL);

        result = null;
        if (glsl && wgsl) {
            const filtered = new ShaderChunks();
            filtered.version = chunks.version;
            chunks.glsl.forEach((code, name) => glsl.has(name) && filtered.glsl.set(name, code));
            chunks.wgsl.forEach((code, name) => wgsl.has(name) && filtered.wgsl.set(name, code));

            // the language of the shader follows from which chunks are overridden, and stays
            const sameLanguage = !device.isWebGPU || filtered.useWGSL === chunks.useWGSL;
            const smaller = filtered.glsl.size < chunks.glsl.size || filtered.wgsl.size < chunks.wgsl.size;
            if (sameLanguage && smaller) {
                result = filtered;
            }
        }
        cache.results.set(key, result);
    }

    return result ?? chunks;
};

export { getPassShaderChunks, reachableChunks };
