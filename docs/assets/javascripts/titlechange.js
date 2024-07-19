document.addEventListener('visibilitychange', function () {
    if (document.visibilityState == 'hidden') {
        normal_title = "欢迎回来!";
        document.title = '网页未响应...';
    } else document.title = normal_title;
});