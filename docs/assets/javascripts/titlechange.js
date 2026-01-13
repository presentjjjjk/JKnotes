document.addEventListener('visibilitychange', function () {
    if (document.visibilityState == 'hidden') {
        normal_title = "JKNOTES:the witness of my learning process";
        document.title = normal_title;
    } else document.title = normal_title;
});