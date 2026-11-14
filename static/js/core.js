window.PlaceIQ = {
  getTheme() {
    return document.documentElement.getAttribute('data-theme') || 'light';
  },
  isDark() {
    return this.getTheme() === 'dark';
  },
  palette() {
    return this.isDark()
      ? ['#3b82f6', '#14b8a6', '#f59e0b', '#8b5cf6', '#fb7185', '#38bdf8', '#34d399']
      : ['#2563eb', '#0d9488', '#d97706', '#7c3aed', '#e11d48', '#0284c7', '#059669'];
  },
  plotlyLayout(extra = {}) {
    const dark = this.isDark();
    const fontColor = dark ? '#cbd5e1' : '#334155';
    const gridColor = dark ? 'rgba(148, 163, 184, 0.12)' : 'rgba(15, 23, 42, 0.08)';
    return Object.assign({
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Plus Jakarta Sans, sans-serif', color: fontColor, size: 12 },
      margin: { t: 28, r: 20, b: 44, l: 52 },
      xaxis: { gridcolor: gridColor, zerolinecolor: gridColor },
      yaxis: { gridcolor: gridColor, zerolinecolor: gridColor },
      legend: { orientation: 'h', y: -0.18 },
    }, extra);
  },
  plotlyConfig() {
    return { responsive: true, displayModeBar: false };
  },
  toggleTheme() {
    const next = this.isDark() ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('placeiq-theme', next);
    const btn = document.getElementById('theme-toggle-btn');
    if (btn) btn.textContent = next === 'dark' ? '☀ Light Mode' : '🌙 Dark Mode';
    if (typeof window.onThemeChanged === 'function') {
      window.onThemeChanged(next);
    }
  },
  initTheme() {
    const saved = localStorage.getItem('placeiq-theme') || 'light';
    document.documentElement.setAttribute('data-theme', saved);
    const btn = document.getElementById('theme-toggle-btn');
    if (btn) btn.textContent = saved === 'dark' ? '☀ Light Mode' : '🌙 Dark Mode';
  },
  readinessBadge(band) {
    if (band === 'Industry Ready') return '<span class="badge badge-green">Industry Ready</span>';
    if (band === 'Developing') return '<span class="badge badge-blue">Developing</span>';
    return '<span class="badge badge-amber">Needs Upskilling</span>';
  }
};

document.addEventListener('DOMContentLoaded', () => {
  window.PlaceIQ.initTheme();
  if (window.lucide) {
    window.lucide.createIcons();
  }
});
