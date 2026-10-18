// Where the loop producers keep their data. The code lives in this repository; the data (job bank,
// state, outputs, ledgers, corpora, sources) does not, and each root can be moved by its variable.
import { homedir } from 'node:os';
import { join } from 'node:path';

const scratch = (name) => join(homedir(), '.claude', 'scratch', name);

export const LANE_HOME = process.env.PLI_QWEN_HOME || scratch('pli-qwen');
export const KB_HOME = process.env.VELD_KB_HOME || scratch('veld-kb');
export const LANG_HOME = process.env.LANG_QWEN_HOME || scratch('lang-qwen');
export const SPINE_REPO = process.env.SPINE_REPO || join(homedir(), 'Repositories', 'Portll', 'spine');
export const Z390_HOME = process.env.Z390 || join(scratch('z390'), 'z390_1.8.4.4');
export const SPINE_PROMPTS_DIR = process.env.SPINE_PROMPTS_DIR || scratch('spine-prompts');
