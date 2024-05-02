const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
//      target: '10.0.1.5:5000',
      target: 'http://localhost:5000',
      changeOrigin: true,
    })
  );
};