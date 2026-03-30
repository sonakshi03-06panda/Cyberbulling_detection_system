/**
 * VibeCheck — main.js
 * Base interactivity: navbar, loading state, flag/hide, animations, counters, contact form.
 */

document.addEventListener('DOMContentLoaded', () => {

  // ── 1. Navbar active state ─────────────────────────────────────
  const currentPath = window.location.pathname.replace(/\/$/, '') || '/';
  document.querySelectorAll('.nav-links a').forEach(link => {
    const linkPath = new URL(link.href).pathname.replace(/\/$/, '') || '/';
    if (linkPath === currentPath) {
      link.classList.add('active');
    }
  });

  // ── 2. Mobile navbar toggle ────────────────────────────────────
  const navToggle = document.getElementById('navToggle');
  const navLinks  = document.getElementById('navLinks');
  if (navToggle && navLinks) {
    navToggle.addEventListener('click', () => {
      navLinks.classList.toggle('open');
      navToggle.textContent = navLinks.classList.contains('open') ? '✕' : '☰';
    });
    // Close on outside click
    document.addEventListener('click', (e) => {
      if (!navToggle.contains(e.target) && !navLinks.contains(e.target)) {
        navLinks.classList.remove('open');
        navToggle.textContent = '☰';
      }
    });
  }

  // ── 3. Analyze form — loading state ───────────────────────────
  const analyzeForm = document.getElementById('analyzeForm');
  const loadingState = document.getElementById('loadingState');
  const submitBtn    = document.getElementById('submitBtn');
  const loadingText  = document.getElementById('loadingText');

  if (analyzeForm && loadingState && submitBtn) {
    const messages = [
      'Fetching comments…',
      'Running AI analysis…',
      'Building your dashboard…',
      'Almost there…',
    ];

    analyzeForm.addEventListener('submit', (e) => {
      const urlInput = document.getElementById('urlInput');
      if (!urlInput || !urlInput.value.trim()) return;

      submitBtn.disabled = true;
      submitBtn.style.opacity = '0.5';
      submitBtn.style.cursor = 'not-allowed';
      analyzeForm.style.display = 'none';
      loadingState.style.display = 'flex';

      let idx = 0;
      if (loadingText) {
        setInterval(() => {
          idx = (idx + 1) % messages.length;
          loadingText.textContent = messages[idx];
        }, 5000);
      }
    });
  }

  // Scroll animations handled by AOS library in base.html

  // ── 5. Animated number counters ───────────────────────────────
  function animateCounter(el, target, duration = 1500) {
    const isFloat = !Number.isInteger(target);
    const decimals = isFloat ? 1 : 0;
    let start = null;

    function step(timestamp) {
      if (!start) start = timestamp;
      const progress = Math.min((timestamp - start) / duration, 1);
      const eased = progress < 0.5
        ? 2 * progress * progress
        : 1 - Math.pow(-2 * progress + 2, 2) / 2;
      const current = target * eased;
      el.textContent = current.toFixed(decimals);
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.textContent = target.toFixed(decimals);
      }
    }

    requestAnimationFrame(step);
  }

  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseFloat(el.getAttribute('data-count'));
    if (!isNaN(target)) {
      el.textContent = '0';
      // Observe counter element; start animation when visible
      if ('IntersectionObserver' in window) {
        const counterObserver = new IntersectionObserver((entries) => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              animateCounter(el, target);
              counterObserver.unobserve(entry.target);
            }
          });
        }, { threshold: 0.5 });
        counterObserver.observe(el);
      } else {
        animateCounter(el, target);
      }
    }
  });

  // ── 6. Auto-dismiss flash messages ────────────────────────────
  document.querySelectorAll('.flash').forEach(flash => {
    setTimeout(() => {
      flash.style.transition = 'opacity 0.4s ease';
      flash.style.opacity = '0';
      setTimeout(() => flash.remove(), 400);
    }, 5000);
  });

  // ── 7. Contact form ────────────────────────────────────────────
  const contactForm    = document.getElementById('contactForm');
  const contactSuccess = document.getElementById('contactSuccess');

  if (contactForm && contactSuccess) {
    contactForm.addEventListener('submit', (e) => {
      e.preventDefault();
      contactForm.style.display = 'none';
      contactSuccess.style.display = 'block';
    });
  }

  // ── 8. Global 3D Parallax (Mouse & Scroll) ─────────────────────────
  const parallaxElements = document.querySelectorAll('.parallax-element');
  if (parallaxElements.length > 0) {
    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;
    let scrollY = window.scrollY;
    let isTicking = false;

    function updateParallax() {
      parallaxElements.forEach(el => {
        const speedX = parseFloat(el.getAttribute('data-speed-x')) || 0;
        const speedY = parseFloat(el.getAttribute('data-speed-y')) || 0;
        
        // Calculate offset based on center of screen for mouse
        const xOffset = (mouseX - window.innerWidth / 2) * speedX;
        const yOffset = (mouseY - window.innerHeight / 2) * speedY;
        
        // Calculate scroll offset (scroll moves things, multiply by speed modifier)
        const scrollOffset = scrollY * (speedY * 2);

        // Apply smooth transition natively or rely on CSS transition if added
        el.style.transform = `translate3d(${xOffset}px, ${yOffset - scrollOffset}px, 0)`;
      });
      isTicking = false;
    }

    document.addEventListener('mousemove', (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
      if (!isTicking) {
        window.requestAnimationFrame(updateParallax);
        isTicking = true;
      }
    });

    document.addEventListener('scroll', () => {
      scrollY = window.scrollY;
      if (!isTicking) {
        window.requestAnimationFrame(updateParallax);
        isTicking = true;
      }
    });
    
    // Initial call
    updateParallax();
  }

});

// ── 9. PDF Export ──────────────────────────────────────────────
window.downloadPDF = function() {
  const element = document.getElementById('report-content');
  if (!element) return;
  
  // Add class to hide UI elements during export
  document.body.classList.add('pdf-exporting');
  
  const opt = {
    margin:       10,
    filename:     'VibeCheck_Report.pdf',
    image:        { type: 'jpeg', quality: 0.98 },
    html2canvas:  { scale: 2, useCORS: true, letterRendering: true, windowWidth: 1200 },
    jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
  };
  
  // Generate PDF
  html2pdf().set(opt).from(element).save().then(() => {
    // Remove class to restore UI elements
    document.body.classList.remove('pdf-exporting');
  });
};
