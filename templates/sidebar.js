const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('mainContent');
const toggleBtn = document.getElementById('toggleBtn');
const overlay = document.getElementById('overlay');

// Load state from localStorage (desktop only)
let isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
let isMobile = window.innerWidth <= 768;

function updateLayout() {
  isMobile = window.innerWidth <= 768;

  if (isMobile) {
    // Mobile: default collapsed, expand with .show
    sidebar.classList.add('collapsed');
    sidebar.classList.remove('show');
    mainContent.classList.add('expanded');
    toggleBtn.classList.add('collapsed-pos');
    overlay.classList.remove('show');
    document.body.classList.remove('sidebar-open');
  } else {
    // Desktop behavior (remember collapse state)
    sidebar.classList.remove('show');
    overlay.classList.remove('show');
    document.body.classList.remove('sidebar-open');

    if (isCollapsed) {
      sidebar.classList.add('collapsed');
      mainContent.classList.add('expanded');
      toggleBtn.classList.add('collapsed-pos');
    } else {
      sidebar.classList.remove('collapsed');
      mainContent.classList.remove('expanded');
      toggleBtn.classList.remove('collapsed-pos');
    }
  }
}

toggleBtn.addEventListener('click', function () {
  if (isMobile) {
    // Mobile toggle
    if (sidebar.classList.contains('show')) {
      // Collapse back
      sidebar.classList.remove('show');
      sidebar.classList.add('collapsed');
      overlay.classList.remove('show');
      toggleBtn.classList.remove('mobile-expanded');
      document.body.classList.remove('sidebar-open');
    } else {
      // Expand
      sidebar.classList.add('show');
      sidebar.classList.remove('collapsed');
      overlay.classList.add('show');
      toggleBtn.classList.add('mobile-expanded');
      document.body.classList.add('sidebar-open');
    }
  } else {
    // Desktop toggle remembers state
    isCollapsed = !isCollapsed;
    localStorage.setItem('sidebarCollapsed', isCollapsed);
    updateLayout();
  }
});

overlay.addEventListener('click', function () {
  if (isMobile) {
    sidebar.classList.remove('show');
    sidebar.classList.add('collapsed');
    overlay.classList.remove('show');
    toggleBtn.classList.remove('mobile-expanded');
    document.body.classList.remove('sidebar-open');
  }
});

window.addEventListener('resize', function() {
  updateLayout();
  // Force height recalculation on resize
  setTimeout(forceSidebarHeight, 50);
});

// Fix height on page load and after navigation
window.addEventListener('load', function() {
  updateLayout();
  setTimeout(forceSidebarHeight, 100);
});

// Fix for navigation changes (if using client-side routing)
window.addEventListener('popstate', function() {
  setTimeout(() => {
    updateLayout();
    forceSidebarHeight();
  }, 10);
});

// Initialize
updateLayout();
setTimeout(forceSidebarHeight, 50);

// Logout function
async function logout() {
  if (confirm('Are you sure you want to logout?')) {
    try {
      const response = await fetch('/api/logout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const data = await response.json();
      if (data.success) {
        window.location.href = '/login';
      } else {
        alert('Error logging out.');
      }
    } catch (error) {
      console.error('Logout error:', error);
      alert('Error logging out.');
    }
  }
}