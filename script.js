// ---- Navbar scroll ----
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

// ---- Mobile menu ----
const navToggle = document.getElementById('navToggle');
const navLinks  = document.getElementById('navLinks');
navToggle.addEventListener('click', () => navLinks.classList.toggle('open'));
document.querySelectorAll('.nav-links a').forEach(a =>
  a.addEventListener('click', () => navLinks.classList.remove('open'))
);

// ---- Scroll fade-in ----
const io = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (e.isIntersecting) { e.target.classList.add('visible'); io.unobserve(e.target); }
  });
}, { threshold: 0.08 });

document.querySelectorAll(
  '.tv2-card, .series-card, .award-v2, .cv2-item, .dig-card, ' +
  '.stat, .style-pillar, .ps-item, .opening-bq'
).forEach(el => { el.classList.add('fade-in'); io.observe(el); });

// ---- Lightbox ----
const lb        = document.getElementById('lightbox');
const lbImg     = lb.querySelector('.lb-img');
const lbCaption = lb.querySelector('.lb-caption');
const lbClose   = lb.querySelector('.lb-close');
const lbBD      = lb.querySelector('.lb-backdrop');

function openLightbox(src, alt, caption) {
  lbImg.src = src;
  lbImg.alt = alt;
  lbCaption.textContent = caption || alt;
  lb.classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  lb.classList.remove('open');
  document.body.style.overflow = '';
}

// Trigger on painting images and dignitary photos
document.querySelectorAll('.ps-image-wrap img, .dignitary-photo, .av2-img-strip img').forEach(img => {
  img.style.cursor = 'zoom-in';
  img.addEventListener('click', () => {
    openLightbox(img.src, img.alt, img.getAttribute('data-caption'));
  });
});

[lbClose, lbBD].forEach(el => el.addEventListener('click', closeLightbox));
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ---- Smooth active nav highlight ----
const sections = document.querySelectorAll('section[id]');
const navAs    = document.querySelectorAll('.nav-links a');

window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 90) current = s.id;
  });
  navAs.forEach(a => {
    a.style.color = a.getAttribute('href') === `#${current}` ? 'var(--teal)' : '';
  });
}, { passive: true });
