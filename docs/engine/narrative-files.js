/**
 * Read narrative text from .txt, .md or .docx in the browser (or Node).
 * .docx uses the vendored JSZip build. .mpp / .pdf are refused with a clear message.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ProjectLensNarrative = factory();
  }
}(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  async function readNarrativeFile(file) {
    const name = (file && file.name) || "narrative";
    const lower = name.toLowerCase();
    if (/\.mpp$/i.test(lower)) {
      throw new Error("Export this schedule from MS Project as XML (File > Save As > XML) and load that");
    }
    if (/\.pdf$/i.test(lower)) {
      throw new Error("Paste the narrative text");
    }
    if (/\.(txt|md)$/i.test(lower) || !/\./.test(lower)) {
      if (typeof file.text === "function") return file.text();
      if (file instanceof Uint8Array) return new TextDecoder("utf-8").decode(file);
      return String(file);
    }
    if (/\.docx$/i.test(lower)) {
      const JSZipRef = typeof JSZip !== "undefined" ? JSZip : null;
      if (!JSZipRef) throw new Error("Word (.docx) narratives need JSZip loaded (docs/vendor/jszip.min.js).");
      const bytes = file.arrayBuffer ? await file.arrayBuffer() : file;
      const zip = await JSZipRef.loadAsync(bytes);
      const doc = zip.file("word/document.xml");
      if (!doc) throw new Error("That .docx file has no word/document.xml.");
      const xml = await doc.async("string");
      // Pull visible text from w:t nodes; keep it crude and deterministic.
      const texts = [...xml.matchAll(/<w:t[^>]*>([^<]*)<\/w:t>/g)].map(m => m[1]);
      return texts.join("").replace(/\s+/g, " ").trim();
    }
    throw new Error("Use a .txt, .md or .docx narrative, or paste the text.");
  }

  return { readNarrativeFile };
}));
