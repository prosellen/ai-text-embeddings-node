import { HfInference } from '@huggingface/inference'
import { config } from 'dotenv';

config({ path: '.env.local' });

const hf = new HfInference(process.env.HF_TOKEN);

const output = await hf.featureExtraction({
  model: "intfloat/e5-small-v2",
  inputs: "That's a happy person"
})

console.log(output)