document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".ajax-form").forEach(form => {
        form.addEventListener("submit", function (e) {
            e.preventDefault();
            const formData = new FormData(this);
            const sendButton = this.querySelector(".send-button");
            const loadingButton = this.querySelector(".loading-button");
            const modalID = this.getAttribute("data-modal");

            sendButton.style.display = "none";
            loadingButton.style.display = "block";

            fetch(this.action, {
                method: "POST",
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => {
                        throw new Error(err.message || 'Произошла ошибка при обработке файла');
                    });
                }
                // Получаем имя файла из заголовка
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = "result"; // Значение по умолчанию

                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/filename="?(.+?)"?(;|$)/i);
                    if (filenameMatch && filenameMatch[1]) {
                        filename = filenameMatch[1];
                    }
                }

                return response.blob().then(blob => ({ blob, filename }));
            })
            .then(({ blob, filename }) => {
                const a = document.createElement("a");
                a.href = URL.createObjectURL(blob);
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);

                sendButton.style.display = "block";
                loadingButton.style.display = "none";
                window.location.reload();
            })
            .catch(error => {
                alert(error.message || "Ошибка обработки файла");
                sendButton.style.display = "block";
                loadingButton.style.display = "none";
            });
        });
    });
}); 