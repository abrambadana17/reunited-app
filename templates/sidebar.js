const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('mainContent');
const toggleBtn = document.getElementById('toggleBtn');
const overlay = document.getElementById('overlay');
let isCollapsed = false;
let isMobile = window.innerWidth <= 768;

function updateLayout() {
  isMobile = window.innerWidth <= 768;

  if (isMobile) {
    sidebar.classList.add('collapsed');
    sidebar.classList.remove('show');
    mainContent.classList.add('expanded');
    toggleBtn.classList.add('collapsed-pos');
    overlay.classList.remove('show');
  } else {
    sidebar.classList.remove('show');
    overlay.classList.remove('show');
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

toggleBtn.addEventListener('click', function() {
  if (isMobile) {
    if (sidebar.classList.contains('show')) {
      sidebar.classList.remove('show');
      overlay.classList.remove('show');
      toggleBtn.classList.remove('mobile-expanded');
    } else {
      sidebar.classList.add('show');
      overlay.classList.add('show');
      toggleBtn.classList.add('mobile-expanded');
    }
  } else {
    isCollapsed = !isCollapsed;
    updateLayout();
  }
});

overlay.addEventListener('click', function() {
  if (isMobile) {
    sidebar.classList.remove('show');
    overlay.classList.remove('show');
    toggleBtn.classList.remove('mobile-expanded');
  }
});

window.addEventListener('resize', updateLayout);
updateLayout();

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
