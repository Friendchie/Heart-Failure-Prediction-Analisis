document.addEventListener("DOMContentLoaded", () => {
    // Obtén referencias a los elementos del DOM
    const toggle = document.querySelector(".toggle");
    const menuDashboard = document.querySelector(".menu-dashboard");
    const iconoMenu = toggle.querySelector("i");
    const enlacesMenu = document.querySelectorAll(".enlace");
    const dashboardContent = document.getElementById("dashboardContent");

    // Función para cargar el contenido del dashboard
    function loadDashboardContent(dashboardNumber) {
        const url = `/static/dashboards/dashboard${dashboardNumber}.html`; // Construye la URL del archivo HTML
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Error al cargar ${url}: ${response.statusText}`);
                }
                return response.text();
            })
            .then(html => {
                console.log('Contenido HTML recibido:', html); // Mensaje para verificar el contenido HTML
                dashboardContent.innerHTML = html;
                dashboardContent.classList.add("open");
            })
            .catch(error => {
                console.error('Error al cargar el contenido del dashboard:', error);
            });
    }

    // Manejador de clic para el menú
    enlacesMenu.forEach(enlace => {
        enlace.addEventListener("click", (e) => {
            e.preventDefault(); // Previene la acción predeterminada del enlace
            dashboardNumber = enlace.getAttribute("data-dashboard");
            loadDashboardContent(dashboardNumber); // Carga el contenido del dashboard
            menuDashboard.classList.add("open");
            iconoMenu.classList.replace("bx-menu", "bx-x");
        });
    });

    // Manejador para el toggle del menú
    toggle.addEventListener("click", () => {
        menuDashboard.classList.toggle("open");

        if (iconoMenu.classList.contains("bx-menu")) {
            iconoMenu.classList.replace("bx-menu", "bx-x");
        } else {
            iconoMenu.classList.replace("bx-x", "bx-menu");
        }
    });
});
