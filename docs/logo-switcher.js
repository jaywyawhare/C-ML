// Conditional logo switcher for Material for MkDocs
(function() {
    'use strict';

    // Function to update logo based on color scheme
    function updateLogo() {
        const logoImg = document.querySelector('.md-header__button.md-logo img');
        if (!logoImg) return;

        // Check current color scheme
        const isDark = document.documentElement.getAttribute('data-md-color-scheme') === 'slate';

        // Update logo source
        // Extract base path from current logo src
        const currentSrc = logoImg.src;
        const basePath = currentSrc.substring(0, currentSrc.lastIndexOf('/') + 1);

        if (isDark) {
            logoImg.src = basePath + 'dark-mode.svg';
        } else {
            logoImg.src = basePath + 'light-mode.svg';
        }
    }

    // Update on page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', updateLogo);
    } else {
        updateLogo();
    }

    // Watch for color scheme changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' && mutation.attributeName === 'data-md-color-scheme') {
                updateLogo();
            }
        });
    });

    // Start observing
    if (document.documentElement) {
        observer.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ['data-md-color-scheme']
        });
    }
})();
