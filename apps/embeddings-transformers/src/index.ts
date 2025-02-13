import { pipeline } from '@huggingface/transformers';

const generateEmbeddings = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2'); // Model is automatically downloaded

function dotProduct(a: number[], b: number[]) {
  if (a.length !== b.length) {
    throw new Error('Both arguments must have the same length');
  }

  let result = 0;

  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }

  return result;
}

const output1 = await generateEmbeddings('This is an example sentence', { pooling: 'mean', normalize: true });
const output2 = await generateEmbeddings('Each sentence is converted', { pooling: 'mean', normalize: true });

console.log(dotProduct(output1.data, output2.data)); // 0.9999999999999999