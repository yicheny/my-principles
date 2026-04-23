export function getExcerpt(body: string | undefined, maxLength = 80): string {
  const text = (body ?? '')
    .replace(/^#{1,6}\s+/gm, '')       // headings
    .replace(/\*\*(.+?)\*\*/g, '$1')   // bold
    .replace(/\*(.+?)\*/g, '$1')       // italic
    .replace(/`(.+?)`/g, '$1')         // inline code
    .replace(/\[(.+?)\]\(.+?\)/g, '$1') // links
    .replace(/!\[.*?\]\(.+?\)/g, '')   // images
    .replace(/>\s?/gm, '')             // blockquotes
    .replace(/[-*+]\s/gm, '')          // list markers
    .replace(/\d+\.\s/gm, '')          // ordered list markers
    .replace(/---+/g, '')              // hr
    .replace(/\n+/g, ' ')             // newlines to spaces
    .trim();

  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '…';
}
