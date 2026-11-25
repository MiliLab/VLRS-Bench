const copyButton = document.querySelector('[data-copy="bibtex"]');

if (copyButton) {
  copyButton.addEventListener('click', async () => {
    const citation = document.querySelector('#bibtex code')?.textContent;
    if (!citation) return;

    try {
      await navigator.clipboard.writeText(citation);
      copyButton.textContent = 'Copied!';
      window.setTimeout(() => { copyButton.textContent = 'Copy BibTeX'; }, 2000);
    } catch {
      copyButton.textContent = 'Select text to copy';
      document.getElementById('bibtex')?.focus();
    }
  });
}
