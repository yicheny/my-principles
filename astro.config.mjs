// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';

/**
 * Vite plugin: fix nested list indentation inside <Fragment> in MDX.
 * MDX strips common indentation from JSX children, leaving sub-items
 * with only 2-space indent — CommonMark needs 3+ for nested ordered lists.
 * This adds 2 extra spaces to sub-indented list items before MDX parses them.
 */
function fixNestedLists() {
  const listLineRe = /^(\s*)(\d+\.|[-*+])\s/;

  function fixIndentation(content) {
    const lines = content.split('\n');
    let minIndent = Infinity;
    let hasList = false;

    for (const line of lines) {
      const m = line.match(listLineRe);
      if (m) { hasList = true; minIndent = Math.min(minIndent, m[1].length); }
    }
    if (!hasList) return content;

    return lines.map(line => {
      const m = line.match(listLineRe);
      if (m && m[1].length > minIndent) return '  ' + line;
      return line;
    }).join('\n');
  }

  return {
    name: 'fix-mdx-nested-lists',
    enforce: /** @type {const} */ ('pre'),
    transform(code, id) {
      if (!id.endsWith('.mdx')) return;
      const result = code.replace(
        /(<Fragment[^>]*>)([\s\S]*?)(<\/Fragment>)/g,
        (_, open, content, close) => open + fixIndentation(content) + close,
      );
      if (result !== code) return { code: result, map: null };
    },
  };
}

// https://astro.build/config
export default defineConfig({
  site: 'https://yicheny.github.io',
  base: '/my-principles',
  integrations: [mdx()],
  vite: { plugins: [fixNestedLists()] },
});
