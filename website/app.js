const toast = document.getElementById('toast');
const menuButton = document.getElementById('menu-button');
const sidebar = document.getElementById('sidebar');

menuButton?.addEventListener('click', () => sidebar.classList.toggle('open'));
document.querySelectorAll('#toc a').forEach(link => link.addEventListener('click', () => sidebar.classList.remove('open')));

document.querySelectorAll('.copy-button').forEach(button => {
  button.addEventListener('click', async () => {
    const text = button.closest('.code-block').querySelector('code').textContent;
    try { await navigator.clipboard.writeText(text); } catch { return; }
    button.textContent = 'Copied';
    toast.classList.add('show');
    setTimeout(() => { button.textContent = 'Copy'; toast.classList.remove('show'); }, 1300);
  });
});

const sections = [...document.querySelectorAll('main section[id]')];
const links = [...document.querySelectorAll('#toc a')];
const observer = new IntersectionObserver(entries => {
  const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (!visible) return;
  links.forEach(link => link.classList.toggle('active', link.getAttribute('href') === `#${visible.target.id}`));
}, { rootMargin: '-18% 0px -72% 0px', threshold: [0, .25, .75] });
sections.forEach(section => observer.observe(section));

const search = document.getElementById('search-input');
function clearMatches() { document.querySelectorAll('.search-match').forEach(item => item.classList.remove('search-match')); }
function runSearch(value) {
  clearMatches();
  const query = value.trim().toLowerCase();
  if (!query) return;
  const match = sections.find(section => section.innerText.toLowerCase().includes(query));
  if (match) { match.classList.add('search-match'); match.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
}
search?.addEventListener('input', event => runSearch(event.target.value));
document.addEventListener('keydown', event => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); search?.focus(); }
  if (event.key === 'Escape' && search) { search.value = ''; clearMatches(); search.blur(); }
});
