// GPT-2 byte-level BPE, matching the released Hugging Face tokenizer.
export class MotionTokenizer {
  constructor(json, tokens) {
    this.vocab = tokens;
    this.inverse = Object.fromEntries(
      Object.entries(tokens).map(([k, v]) => [v, k]),
    );
    this.ranks = new Map(
      json.model.merges.map((pair, i) => [
        Array.isArray(pair) ? pair.join(" ") : pair,
        i,
      ]),
    );
    let bytes = [];
    for (let i = 33; i <= 126; i++) bytes.push(i);
    for (let i = 161; i <= 172; i++) bytes.push(i);
    for (let i = 174; i <= 255; i++) bytes.push(i);
    let chars = bytes.slice(),
      extra = 0;
    for (let i = 0; i < 256; i++)
      if (!bytes.includes(i)) {
        bytes.push(i);
        chars.push(256 + extra++);
      }
    this.byteEncoder = Object.fromEntries(
      bytes.map((b, i) => [b, String.fromCodePoint(chars[i])]),
    );
    this.byteDecoder = Object.fromEntries(
      bytes.map((b, i) => [String.fromCodePoint(chars[i]), b]),
    );
    this.cache = new Map();
    this.special = new Set(json.added_tokens.map((x) => x.content));
    this.split = new RegExp(
      "(" +
        [...this.special]
          .sort((a, b) => b.length - a.length)
          .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
          .join("|") +
        ")",
      "g",
    );
  }
  bpe(word) {
    if (this.cache.has(word)) return this.cache.get(word);
    let pieces = Array.from(word);
    while (pieces.length > 1) {
      let index = -1,
        rank = Infinity;
      for (let i = 0; i < pieces.length - 1; i++) {
        let r = this.ranks.get(pieces[i] + " " + pieces[i + 1]);
        if (r !== undefined && r < rank) {
          rank = r;
          index = i;
        }
      }
      if (index < 0) break;
      const left = pieces[index],
        right = pieces[index + 1],
        merged = [];
      for (let i = 0; i < pieces.length; i++) {
        if (pieces[i] === left && pieces[i + 1] === right) {
          merged.push(left + right);
          i++;
        } else merged.push(pieces[i]);
      }
      pieces = merged;
    }
    this.cache.set(word, pieces);
    return pieces;
  }
  encode(text) {
    const ids = [];
    for (const part of text.split(this.split)) {
      if (!part) continue;
      if (this.special.has(part)) {
        ids.push(this.vocab[part]);
        continue;
      }
      const chunks =
        part.match(
          /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu,
        ) || [];
      for (const chunk of chunks) {
        const word = Array.from(
          new TextEncoder().encode(chunk),
          (b) => this.byteEncoder[b],
        ).join("");
        for (const token of this.bpe(word)) {
          const id = this.vocab[token];
          if (id === undefined)
            throw Error("Tokenizer could not encode input.");
          ids.push(id);
        }
      }
    }
    return ids;
  }
  decode(ids) {
    let output = "",
      bytes = [];
    const flush = () => {
      output += new TextDecoder().decode(new Uint8Array(bytes));
      bytes = [];
    };
    for (const id of ids) {
      const token = this.inverse[id];
      if (this.special.has(token)) {
        flush();
        output += token;
      } else for (const c of token || "") bytes.push(this.byteDecoder[c]);
    }
    flush();
    return output;
  }
}
