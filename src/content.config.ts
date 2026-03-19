import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const principleSchema = z.object({
  title: z.string(),
  order: z.number().default(0),
  tags: z.array(z.string()).default([]),
  updatedAt: z.string().optional(),
});

const dailySchema = z.object({
  title: z.string(),
  date: z.string(),
  tags: z.array(z.string()).default([]),
});

const learn = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/learn' }),
  schema: principleSchema,
});

const principles = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/principles' }),
  schema: principleSchema,
});

const daily = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/daily' }),
  schema: dailySchema,
});

export const collections = { learn, principles, daily };
