(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[2936],{43894:function(e,n,t){"use strict";t.d(n,{zx:function(){return w},hU:function(){return A}});var r=t(97375),o=t(11418),a=t(38554),i=t.n(a);t(41706);var u=!1;!function(e){var n=new WeakMap}((function(e,n,t,r){var o="string"===typeof n?n.split("."):[n];for(r=0;r<o.length&&e;r+=1)e=e[o[r]];return void 0===e?t:e}));"undefined"===typeof window||!window.document||window.document.createElement;var l=function(e){return e?"":void 0},c=function(){for(var e=arguments.length,n=new Array(e),t=0;t<e;t++)n[t]=arguments[t];return n.filter(Boolean).join(" ")};["input:not([disabled])","select:not([disabled])","textarea:not([disabled])","embed","iframe","object","a[href]","area[href]","button:not([disabled])","[tabindex]","audio[controls]","video[controls]","*[tabindex]:not([aria-disabled])","*[contenteditable]"].join();function f(e){var n;return function(){if(e){for(var t=arguments.length,r=new Array(t),o=0;o<t;o++)r[o]=arguments[o];n=e.apply(this,r),e=null}return n}}f((function(e){return function(){e.condition,e.message}})),f((function(e){return function(){e.condition,e.message}}));Number.MIN_SAFE_INTEGER,Number.MAX_SAFE_INTEGER;Object.freeze(["base","sm","md","lg","xl","2xl"]);var s=t(67294),d=t(8922),p=t(44359);function v(e,n){if(null==e)return{};var t,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)t=a[r],n.indexOf(t)>=0||(o[t]=e[t]);return o}function m(){return m=Object.assign||function(e){for(var n=1;n<arguments.length;n++){var t=arguments[n];for(var r in t)Object.prototype.hasOwnProperty.call(t,r)&&(e[r]=t[r])}return e},m.apply(this,arguments)}var h=(0,d.kr)({strict:!1,name:"ButtonGroupContext"}),g=(h[0],h[1]);var b=["label","placement","spacing","children","className","__css"],y=function(e){var n=e.label,t=e.placement,r=e.spacing,a=void 0===r?"0.5rem":r,i=e.children,u=void 0===i?s.createElement(p.$,{color:"currentColor",width:"1em",height:"1em"}):i,l=e.className,f=e.__css,d=v(e,b),h=c("chakra-button__spinner",l),g="start"===t?"marginEnd":"marginStart",y=s.useMemo((function(){var e;return m(((e={display:"flex",alignItems:"center",position:n?"relative":"absolute"})[g]=n?a:0,e.fontSize="1em",e.lineHeight="normal",e),f)}),[f,n,g,a]);return s.createElement(o.m$.div,m({className:h},d,{__css:y}),u)};var E=["children","className"],x=function(e){var n=e.children,t=e.className,r=v(e,E),a=s.isValidElement(n)?s.cloneElement(n,{"aria-hidden":!0,focusable:!1}):n,i=c("chakra-button__icon",t);return s.createElement(o.m$.span,m({display:"inline-flex",alignSelf:"center",flexShrink:0},r,{className:i}),a)};var _=["isDisabled","isLoading","isActive","isFullWidth","children","leftIcon","rightIcon","loadingText","iconSpacing","type","spinner","spinnerPlacement","className","as"],w=(0,o.Gp)((function(e,n){var t=g(),a=(0,o.mq)("Button",m({},t,e)),u=(0,o.Lr)(e),f=u.isDisabled,d=void 0===f?null==t?void 0:t.isDisabled:f,p=u.isLoading,h=u.isActive,b=u.isFullWidth,E=u.children,x=u.leftIcon,w=u.rightIcon,S=u.loadingText,A=u.iconSpacing,N=void 0===A?"0.5rem":A,k=u.type,j=u.spinner,M=u.spinnerPlacement,C=void 0===M?"start":M,L=u.className,R=u.as,O=v(u,_),T=s.useMemo((function(){var e,n=i()({},null!=(e=null==a?void 0:a._focus)?e:{},{zIndex:1});return m({display:"inline-flex",appearance:"none",alignItems:"center",justifyContent:"center",userSelect:"none",position:"relative",whiteSpace:"nowrap",verticalAlign:"middle",outline:"none",width:b?"100%":"auto"},a,!!t&&{_focus:n})}),[a,t,b]),U=function(e){var n=s.useState(!e),t=n[0],r=n[1];return{ref:s.useCallback((function(e){e&&r("BUTTON"===e.tagName)}),[]),type:t?"button":void 0}}(R),$=U.ref,D=U.type,P={rightIcon:w,leftIcon:x,iconSpacing:N,children:E};return s.createElement(o.m$.button,m({disabled:d||p,ref:(0,r.qq)(n,$),as:R,type:null!=k?k:D,"data-active":l(h),"data-loading":l(p),__css:T,className:c("chakra-button",L)},O),p&&"start"===C&&s.createElement(y,{className:"chakra-button__spinner--start",label:S,placement:"start",spacing:N},j),p?S||s.createElement(o.m$.span,{opacity:0},s.createElement(I,P)):s.createElement(I,P),p&&"end"===C&&s.createElement(y,{className:"chakra-button__spinner--end",label:S,placement:"end",spacing:N},j))}));function I(e){var n=e.leftIcon,t=e.rightIcon,r=e.children,o=e.iconSpacing;return s.createElement(s.Fragment,null,n&&s.createElement(x,{marginEnd:o},n),r,t&&s.createElement(x,{marginStart:o},t))}var S=["icon","children","isRound","aria-label"],A=(0,o.Gp)((function(e,n){var t=e.icon,r=e.children,o=e.isRound,a=e["aria-label"],i=v(e,S),u=t||r,l=s.isValidElement(u)?s.cloneElement(u,{"aria-hidden":!0,focusable:!1}):null;return s.createElement(w,m({padding:"0",borderRadius:o?"full":void 0,ref:n,"aria-label":a},i),l)}))},44286:function(e){e.exports=function(e){return e.split("")}},41848:function(e){e.exports=function(e,n,t,r){for(var o=e.length,a=t+(r?1:-1);r?a--:++a<o;)if(n(e[a],a,e))return a;return-1}},42118:function(e,n,t){var r=t(41848),o=t(62722),a=t(42351);e.exports=function(e,n,t){return n===n?a(e,n,t):r(e,o,t)}},62722:function(e){e.exports=function(e){return e!==e}},14259:function(e){e.exports=function(e,n,t){var r=-1,o=e.length;n<0&&(n=-n>o?0:o+n),(t=t>o?o:t)<0&&(t+=o),o=n>t?0:t-n>>>0,n>>>=0;for(var a=Array(o);++r<o;)a[r]=e[r+n];return a}},40180:function(e,n,t){var r=t(14259);e.exports=function(e,n,t){var o=e.length;return t=void 0===t?o:t,!n&&t>=o?e:r(e,n,t)}},5512:function(e,n,t){var r=t(42118);e.exports=function(e,n){for(var t=e.length;t--&&r(n,e[t],0)>-1;);return t}},62689:function(e){var n=RegExp("[\\u200d\\ud800-\\udfff\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff\\ufe0e\\ufe0f]");e.exports=function(e){return n.test(e)}},42351:function(e){e.exports=function(e,n,t){for(var r=t-1,o=e.length;++r<o;)if(e[r]===n)return r;return-1}},83140:function(e,n,t){var r=t(44286),o=t(62689),a=t(676);e.exports=function(e){return o(e)?a(e):r(e)}},67990:function(e){var n=/\s/;e.exports=function(e){for(var t=e.length;t--&&n.test(e.charAt(t)););return t}},676:function(e){var n="[\\ud800-\\udfff]",t="[\\u0300-\\u036f\\ufe20-\\ufe2f\\u20d0-\\u20ff]",r="\\ud83c[\\udffb-\\udfff]",o="[^\\ud800-\\udfff]",a="(?:\\ud83c[\\udde6-\\uddff]){2}",i="[\\ud800-\\udbff][\\udc00-\\udfff]",u="(?:"+t+"|"+r+")"+"?",l="[\\ufe0e\\ufe0f]?",c=l+u+("(?:\\u200d(?:"+[o,a,i].join("|")+")"+l+u+")*"),f="(?:"+[o+t+"?",t,a,i,n].join("|")+")",s=RegExp(r+"(?="+r+")|"+f+c,"g");e.exports=function(e){return e.match(s)||[]}},48418:function(e,n,t){"use strict";function r(e,n){(null==n||n>e.length)&&(n=e.length);for(var t=0,r=new Array(n);t<n;t++)r[t]=e[t];return r}function o(e,n){return function(e){if(Array.isArray(e))return e}(e)||function(e,n){var t=null==e?null:"undefined"!==typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(null!=t){var r,o,a=[],i=!0,u=!1;try{for(t=t.call(e);!(i=(r=t.next()).done)&&(a.push(r.value),!n||a.length!==n);i=!0);}catch(l){u=!0,o=l}finally{try{i||null==t.return||t.return()}finally{if(u)throw o}}return a}}(e,n)||function(e,n){if(!e)return;if("string"===typeof e)return r(e,n);var t=Object.prototype.toString.call(e).slice(8,-1);"Object"===t&&e.constructor&&(t=e.constructor.name);if("Map"===t||"Set"===t)return Array.from(t);if("Arguments"===t||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t))return r(e,n)}(e,n)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}n.default=void 0;var a,i=(a=t(67294))&&a.__esModule?a:{default:a},u=t(76273),l=t(90387),c=t(57190);var f={};function s(e,n,t,r){if(e&&u.isLocalURL(n)){e.prefetch(n,t,r).catch((function(e){0}));var o=r&&"undefined"!==typeof r.locale?r.locale:e&&e.locale;f[n+"%"+t+(o?"%"+o:"")]=!0}}var d=function(e){var n,t=!1!==e.prefetch,r=l.useRouter(),a=i.default.useMemo((function(){var n=o(u.resolveHref(r,e.href,!0),2),t=n[0],a=n[1];return{href:t,as:e.as?u.resolveHref(r,e.as):a||t}}),[r,e.href,e.as]),d=a.href,p=a.as,v=e.children,m=e.replace,h=e.shallow,g=e.scroll,b=e.locale;"string"===typeof v&&(v=i.default.createElement("a",null,v));var y=(n=i.default.Children.only(v))&&"object"===typeof n&&n.ref,E=o(c.useIntersection({rootMargin:"200px"}),2),x=E[0],_=E[1],w=i.default.useCallback((function(e){x(e),y&&("function"===typeof y?y(e):"object"===typeof y&&(y.current=e))}),[y,x]);i.default.useEffect((function(){var e=_&&t&&u.isLocalURL(d),n="undefined"!==typeof b?b:r&&r.locale,o=f[d+"%"+p+(n?"%"+n:"")];e&&!o&&s(r,d,p,{locale:n})}),[p,d,_,b,t,r]);var I={ref:w,onClick:function(e){n.props&&"function"===typeof n.props.onClick&&n.props.onClick(e),e.defaultPrevented||function(e,n,t,r,o,a,i,l){("A"!==e.currentTarget.nodeName.toUpperCase()||!function(e){var n=e.currentTarget.target;return n&&"_self"!==n||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey||e.nativeEvent&&2===e.nativeEvent.which}(e)&&u.isLocalURL(t))&&(e.preventDefault(),n[o?"replace":"push"](t,r,{shallow:a,locale:l,scroll:i}))}(e,r,d,p,m,h,g,b)},onMouseEnter:function(e){n.props&&"function"===typeof n.props.onMouseEnter&&n.props.onMouseEnter(e),u.isLocalURL(d)&&s(r,d,p,{priority:!0})}};if(e.passHref||"a"===n.type&&!("href"in n.props)){var S="undefined"!==typeof b?b:r&&r.locale,A=r&&r.isLocaleDomain&&u.getDomainLocale(p,S,r&&r.locales,r&&r.domainLocales);I.href=A||u.addBasePath(u.addLocale(p,S,r&&r.defaultLocale))}return i.default.cloneElement(n,I)};n.default=d},57190:function(e,n,t){"use strict";function r(e,n){(null==n||n>e.length)&&(n=e.length);for(var t=0,r=new Array(n);t<n;t++)r[t]=e[t];return r}function o(e,n){return function(e){if(Array.isArray(e))return e}(e)||function(e,n){var t=null==e?null:"undefined"!==typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(null!=t){var r,o,a=[],i=!0,u=!1;try{for(t=t.call(e);!(i=(r=t.next()).done)&&(a.push(r.value),!n||a.length!==n);i=!0);}catch(l){u=!0,o=l}finally{try{i||null==t.return||t.return()}finally{if(u)throw o}}return a}}(e,n)||function(e,n){if(!e)return;if("string"===typeof e)return r(e,n);var t=Object.prototype.toString.call(e).slice(8,-1);"Object"===t&&e.constructor&&(t=e.constructor.name);if("Map"===t||"Set"===t)return Array.from(t);if("Arguments"===t||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(t))return r(e,n)}(e,n)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}Object.defineProperty(n,"__esModule",{value:!0}),n.useIntersection=function(e){var n=e.rootRef,t=e.rootMargin,r=e.disabled||!u,f=a.useRef(),s=o(a.useState(!1),2),d=s[0],p=s[1],v=o(a.useState(n?n.current:null),2),m=v[0],h=v[1],g=a.useCallback((function(e){f.current&&(f.current(),f.current=void 0),r||d||e&&e.tagName&&(f.current=function(e,n,t){var r=function(e){var n,t={root:e.root||null,margin:e.rootMargin||""},r=c.find((function(e){return e.root===t.root&&e.margin===t.margin}));r?n=l.get(r):(n=l.get(t),c.push(t));if(n)return n;var o=new Map,a=new IntersectionObserver((function(e){e.forEach((function(e){var n=o.get(e.target),t=e.isIntersecting||e.intersectionRatio>0;n&&t&&n(t)}))}),e);return l.set(t,n={id:t,observer:a,elements:o}),n}(t),o=r.id,a=r.observer,i=r.elements;return i.set(e,n),a.observe(e),function(){if(i.delete(e),a.unobserve(e),0===i.size){a.disconnect(),l.delete(o);var n=c.findIndex((function(e){return e.root===o.root&&e.margin===o.margin}));n>-1&&c.splice(n,1)}}}(e,(function(e){return e&&p(e)}),{root:m,rootMargin:t}))}),[r,m,t,d]);return a.useEffect((function(){if(!u&&!d){var e=i.requestIdleCallback((function(){return p(!0)}));return function(){return i.cancelIdleCallback(e)}}}),[d]),a.useEffect((function(){n&&h(n.current)}),[n]),[g,d]};var a=t(67294),i=t(9311),u="undefined"!==typeof IntersectionObserver;var l=new Map,c=[]},41664:function(e,n,t){e.exports=t(48418)}}]);