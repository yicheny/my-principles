import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const principleSchema = z.object({
  title: z.string(),
  order: z.number().default(0),
  tags: z.array(z.string()).default([]),
  updatedAt: z.string().optional(),
});

const work = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/work' }),
  schema: principleSchema,
});

const life = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/life' }),
  schema: principleSchema,
});

export const collections = { work, life };
