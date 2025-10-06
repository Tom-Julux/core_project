export default {
  title: 'Napari Plugins for Interactive Segmentation',
  description: 'User guide for interactive segmentation using Napari plugins.',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Docs', link: '/getting_started/' },
      { text: 'Reference', link: '/reference/' },
    ],
    search:{
      provider: 'local'
    },
    sidebar: [
      {
        text: 'User Guide',
        items: [
          { text: 'Introduction', link: '/user-guide/introduction' },
          { text: 'Base Widget UI', link: '/user-guide/base-widget-ui' }
        ]
      },
      {
        text: 'Getting Started',
        items: [
          { text: 'First Steps', link: '/getting_started/first_steps' },
          { text: 'Installation', link: '/getting_started/installation' }
        ]
      },
      {
        text: 'Reference',
        items: [
            { text: 'Overview', link: '/reference' }, 
          { text: 'Windowing', link: '/reference/windowing' },
          { text: 'Multi-Object Mode', link: '/reference/multi-object-mode' }
        ]
      },
      {
        text: 'Roadmap',
        items: [
          { text: 'Roadmap', link: '/roadmap' }
        ]
      }
    ],
     footer: {
      message: 'Released under the <a href="https://github.com/vuejs/vitepress/blob/main/LICENSE">MIT License</a>.',
      copyright: 'Copyright © 2019-present <a href="https://github.com/yyx990803">Evan You</a>'
    }
  }
};