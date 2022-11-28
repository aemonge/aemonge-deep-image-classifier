"use strict";(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[4107],{71015:function(e,n,r){r.d(n,{M:function(){return o},O:function(){return i}});var t=r(67294),o=t.createContext(null),i=function(){var e=t.useContext(o);if(!e)throw new Error("Atom compound components cannot be rendered outside the Atom component");return e}},98405:function(e,n,r){r.d(n,{l:function(){return u}});var t=r(85893),o=(r(67294),r(11163)),i=r(82954),l=r(63598),u=function(e){var n=e.paramName,r=(0,o.useRouter)(),u=n?r.query[n]:void 0,s=(0,l.r)().requestedContentKeys;return u&&s[u]?(console.error("Attempted to access content key: ".concat(u)),r.push("/my-programs"),(0,t.jsx)(i.S,{})):(0,t.jsx)(i.S,{})}},92710:function(e,n,r){r.d(n,{I:function(){return u}});var t=r(85893),o=(r(67294),r(27634)),i=r(26707),l=r(98388),u=function(e){var n=e.message;return(0,t.jsx)(o.J1,{children:(0,t.jsxs)(i.bZ,{status:"error",variant:"embedded",children:[(0,t.jsx)(l.aNP,{w:8,h:8,color:"red"}),(0,t.jsx)(i.X,{children:n})]})})}},58013:function(e,n,r){r.d(n,{u:function(){return s}});var t=r(85893),o=(r(67294),r(26793)),i=r(83580),l=r(20835),u=r(23256),s=function(e){var n=e.isOpen,r=e.onClose,s=e.title,a=e.titleSize,c=e.bodyStyles,d=e.footer,f=e.footerStyles,v=e.headerHidden,p=e.label,h=e.closable,m=void 0===h||h,b=e.children,x=(0,o.$)(),y=x.i18n,j=x.t,g="rtl"===y.dir(),w=(0,i.Sx)({base:"mobile",md:"tablet",lg:"desktop"});return(0,t.jsxs)(l.u_,{isOpen:n,onClose:r,isCentered:!0,size:w,closeOnOverlayClick:m,children:[(0,t.jsx)(l.ZA,{}),(0,t.jsxs)(l.hz,{p:{base:8,lg:12},"aria-label":p,children:[s&&(0,t.jsx)(l.xB,{hidden:v,children:(0,t.jsx)(u.X6,{textAlign:"center",size:null!==a&&void 0!==a?a:"h2",children:s})}),m&&(0,t.jsx)(l.ol,{color:"silver",size:"lg","aria-label":j("common.closeModal"),borderRadius:"circle",sx:g?{left:3,right:"unset"}:{}}),(0,t.jsx)(l.fe,{p:0,sx:c,children:b}),d&&(0,t.jsx)(l.mz,{sx:f,children:d})]})]})}},92481:function(e,n,r){r.d(n,{_:function(){return s}});var t=r(85893),o=(r(67294),r(11163)),i=r(98405),l=r(60299),u=r(63598),s=function(e){var n=e.children,r=(0,o.useRouter)(),s=r.query,a=s.nanodegreeKey,c=s.freeCourseKey,d=s.paidCourseKey,f=(0,u.r)(),v=f.nanodegree,p=f.freeCourse,h=f.paidCourse;if(!f.lesson){if(!v&&!p&&!h){var m=a?"nanodegreeKey":c?"freeCourseKey":d?"paidCourseKey":null;if(null!==m)return(0,t.jsx)(i.l,{paramName:m})}v?r.push(l.F.nanodegree(v)):h?r.push(l.F.paidCourse(h)):p&&r.push(l.F.freeCourse(p))}return(0,t.jsx)(t.Fragment,{children:n})}},22366:function(e,n,r){r.d(n,{Mq:function(){return l},aO:function(){return u},n2:function(){return s}});var t=r(26367);function o(e,n){(null==n||n>e.length)&&(n=e.length);for(var r=0,t=new Array(n);r<n;r++)t[r]=e[r];return t}function i(e){return function(e){if(Array.isArray(e))return o(e)}(e)||function(e){if("undefined"!==typeof Symbol&&null!=e[Symbol.iterator]||null!=e["@@iterator"])return Array.from(e)}(e)||function(e,n){if(!e)return;if("string"===typeof e)return o(e,n);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(r);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return o(e,n)}(e)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}var l=function(e){return!!(null===e||void 0===e?void 0:e.skillConfidenceRatingAfter)},u=function(e){return e.state===t.kA.Passed},s=function(e,n){var r=i(n.pathname.split("/")).pop();if(!r)throw new Error("No trailingSegment found");return"[labKey]"===r?"welcome"===e?n.asPath:"".concat(n.asPath,"/").concat(e):"welcome"===e?n.asPath.replace(r,""):n.asPath.replace(r,e)}},33725:function(e,n,r){r.d(n,{Z:function(){return te}});var t,o=r(85893),i=r(67294),l=r(26793),u=r(97375),s=r(23256),a=r(43894),c=r(98388),d=r(71015),f=r(17570),v=r(70641),p=r(71447),h=r(63598),m=r(86190);!function(e){e.FILE="file"}(t||(t={}));var b=r(14416),x=r(44359),y=r(27634),j=r(30268),g=r(86211),w=r(48125),C=r(92710),k=r(58013),S=r(7484);function O(e,n){(null==n||n>e.length)&&(n=e.length);for(var r=0,t=new Array(n);r<n;r++)t[r]=e[r];return t}function I(e,n){return function(e){if(Array.isArray(e))return e}(e)||function(e,n){var r=null==e?null:"undefined"!==typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(null!=r){var t,o,i=[],l=!0,u=!1;try{for(r=r.call(e);!(l=(t=r.next()).done)&&(i.push(t.value),!n||i.length!==n);l=!0);}catch(s){u=!0,o=s}finally{try{l||null==r.return||r.return()}finally{if(u)throw o}}return i}}(e,n)||function(e,n){if(!e)return;if("string"===typeof e)return O(e,n);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(r);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return O(e,n)}(e,n)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}var R=(0,r(22222).createSelector)((function(e){return e.cloudResource}),(function(e,n,r){return[n,r]}),(function(e,n){return e.find((function(e){var r=I(e,2),t=I(r[0],2),o=t[0],i=t[1];r[1];return o===n[0]&&i===n[1]}))})),A=r(428),z=r(1921);function P(e,n){(null==n||n>e.length)&&(n=e.length);for(var r=0,t=new Array(n);r<n;r++)t[r]=e[r];return t}function _(e,n){return function(e){if(Array.isArray(e))return e}(e)||function(e,n){var r=null==e?null:"undefined"!==typeof Symbol&&e[Symbol.iterator]||e["@@iterator"];if(null!=r){var t,o,i=[],l=!0,u=!1;try{for(r=r.call(e);!(l=(t=r.next()).done)&&(i.push(t.value),!n||i.length!==n);l=!0);}catch(s){u=!0,o=s}finally{try{l||null==r.return||r.return()}finally{if(u)throw o}}return i}}(e,n)||function(e,n){if(!e)return;if("string"===typeof e)return P(e,n);var r=Object.prototype.toString.call(e).slice(8,-1);"Object"===r&&e.constructor&&(r=e.constructor.name);if("Map"===r||"Set"===r)return Array.from(r);if("Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r))return P(e,n)}(e,n)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}()}var T=function(){var e=(0,S.t)(),n=e.serviceId,r=e.enrollmentId;return n?(0,o.jsx)(M,{serviceId:n,enrollmentId:r}):null},M=function(e){var n=e.serviceId,r=e.enrollmentId,t=(0,i.useState)(!1),u=t[0],d=t[1],f=(0,i.useState)(!1),v=f[0],p=f[1],h=(0,i.useState)(""),m=h[0],x=h[1],y=(0,i.useState)(null),j=y[0],g=y[1],w=(0,l.$)().t,C=(0,b.useDispatch)(),S=(0,b.useSelector)((function(e){return R(e,n,r)})),O=function(){d(!0),I()},I=function(){var e=S&&(0,z.L)(S[1]);if(!S||e)p(!0),x(""),(0,z.n)(n,r).then((function(e){var t=e.data.resource;g(t),C((0,A.$o)({serviceId:n,resource:t,enrollmentId:r}))})).catch((function(e){console.error(e),x(w("cloudResources.error"))})).finally((function(){return p(!1)}));else{var t=_(S,2)[1];g(t)}};return(0,o.jsxs)(s.kC,{justifyContent:"space-between",alignItems:"center",onClick:O,p:1,ps:4,zIndex:"low",bg:"blue-dark",boxShadow:"shadow-1",borderBottom:"1px solid 'slate-darker'",borderTop:"1px solid 'slate-darker'",mb:"2px",children:[(0,o.jsx)(s.X6,{size:"h5",color:"white",children:w("cloudResources.launchCloud")}),(0,o.jsx)(a.hU,{"aria-label":w("cloudResources.launchCloud"),icon:(0,o.jsx)(c.aWb,{color:"white",w:8,h:8}),onClick:O,variant:"minimal-inverse",isRound:!0,p:2}),(0,o.jsx)(k.u,{"aria-label":w("cloudResources.launchCloud"),isOpen:u,onClose:function(){d(!1)},children:(0,o.jsx)($,{loading:v,errorMessage:m,resource:j})})]})},E="aws",F="azure",$=function(e){var n=e.loading,r=e.errorMessage,t=e.resource,i=(0,l.$)().t;return n?(0,o.jsx)(s.kC,{justifyContent:"center",children:(0,o.jsx)(x.$,{label:i("common.loading")})}):r?(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(s.X6,{size:"h4",children:i("cloudResources.errorHeading")}),(0,o.jsx)(C.I,{message:r})]}):null===t?null:(0,o.jsxs)(o.Fragment,{children:[t.service===E&&(0,o.jsx)(K,{resource:t}),t.service===F&&(0,o.jsx)(N,{resource:t}),(0,o.jsxs)(w.b,{variant:"primary",href:t.url,children:[(0,o.jsx)(c.xPt,{w:8,h:8})," ",i("cloudResources.cloudOpenConsole")]})]})},K=function(e){var n=e.resource,r=(0,l.$)().t;return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(q,{id:"aws-access-key-id",label:r("cloudResources.awsAccessKeyId"),value:n.aws_access_key_id}),(0,o.jsx)(q,{id:"aws-secret-access-key",label:r("cloudResources.awsSecretAccessKey"),value:n.aws_secret_access_key}),(0,o.jsx)(q,{id:"aws-session-token",label:r("cloudResources.awsSessionToken"),value:n.aws_session_token})]})},N=function(e){var n=e.resource,r=(0,l.$)().t;return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(q,{id:"azure-username",label:r("cloudResources.azureUsername"),value:n.azure_username}),(0,o.jsx)(q,{id:"azure-password",label:r("cloudResources.azurePassword"),value:n.azure_password})]})},q=function(e){var n=e.id,r=e.label,t=e.value;return(0,o.jsx)(s.xu,{mb:6,children:(null!==t&&void 0!==t?t:"").length<75?(0,o.jsxs)(y.NI,{children:[(0,o.jsx)(y.lX,{fontWeight:"semibold",children:r}),(0,o.jsx)(j.II,{id:n,value:t,onChange:function(){},readOnly:!0,required:!0})]}):(0,o.jsxs)(y.NI,{children:[(0,o.jsx)(y.lX,{fontWeight:"semibold",children:r}),(0,o.jsx)(g.g,{id:n,value:t,onChange:function(){},rows:4,readOnly:!0,required:!0})]})})},L=r(22366),X=r(7886),H=r(3397),U=function(e){return e.isCompleted?(0,o.jsx)(c.nQG,{color:"green",w:6,h:6}):(0,o.jsx)(c.TXi,{w:6,h:6})};function B(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function D(e,n){if(null==e)return{};var r,t,o=function(e,n){if(null==e)return{};var r,t,o={},i=Object.keys(e);for(t=0;t<i.length;t++)r=i[t],n.indexOf(r)>=0||(o[r]=e[r]);return o}(e,n);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(t=0;t<i.length;t++)r=i[t],n.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(o[r]=e[r])}return o}var Z=function(e){e.content;var n=e.isActive,r=e.isCompleted,t=e.iconOverride,l=e.label,u=e.onScroll,a=D(e,["content","isActive","isCompleted","iconOverride","label","onScroll"]),c=(0,i.useRef)(null);return(0,i.useEffect)((function(){u&&c.current&&u(c.current)}),[u]),(0,o.jsxs)(s.HC,function(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{},t=Object.keys(r);"function"===typeof Object.getOwnPropertySymbols&&(t=t.concat(Object.getOwnPropertySymbols(r).filter((function(e){return Object.getOwnPropertyDescriptor(r,e).enumerable})))),t.forEach((function(n){B(e,n,r[n])}))}return e}({p:4,display:"grid",gridTemplateColumns:"auto 1fr",alignItems:"center",columnGap:4,cursor:"pointer",color:n?"cerulean":"white"},a,{ref:c,children:[t||(0,o.jsx)(U,{isCompleted:r}),(0,o.jsx)(s.xv,{size:"sm",children:l})]}))},G=function(e){var n=e.activeContent,r=e.labResult,t=e.lesson,i=e.onConceptClick,u=e.onScroll,a=(0,l.$)().t;return t?(0,o.jsx)(s.QI,{m:0,p:0,listStyleType:"none",overflow:"auto",pos:"relative",h:"full",children:(0,X.kz)(t).map((function(e,l){var s=e===n;if((0,H.Op)(e)){var d,f,v,p=e.isSubmitProject?(0,o.jsx)(c.Rg1,{color:(null===(d=t.project)||void 0===d||null===(f=d.projectState)||void 0===f?void 0:f.status)===H.kJ.PASSED?"green":void 0,w:6,h:6}):void 0;return(0,o.jsx)(Z,{content:e,isActive:s,isCompleted:!!(null===(v=e.userState)||void 0===v?void 0:v.completedAt),iconOverride:p,label:"".concat(l+1,". ").concat(e.title),onClick:function(){return i(e)},onScroll:s?u:void 0},e.key)}var h=(0,L.Mq)(r),m=h?void 0:(0,o.jsx)(c.VGR,{w:6,h:6});return(0,o.jsx)(Z,{content:e,iconOverride:m,isActive:e===n,isCompleted:h,label:"".concat(l+1,". ").concat(a("common.lab"),": ").concat(e.title),onClick:function(){return i(e)},onScroll:u},e.key)}))}):(0,o.jsx)(s.QI,{m:0,p:0,listStyleType:"none",overflow:"auto"})};function Q(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function W(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{},t=Object.keys(r);"function"===typeof Object.getOwnPropertySymbols&&(t=t.concat(Object.getOwnPropertySymbols(r).filter((function(e){return Object.getOwnPropertyDescriptor(r,e).enumerable})))),t.forEach((function(n){Q(e,n,r[n])}))}return e}var J=function(e){var n,r=e.onClose,u=(0,h.r)(),d=u.part,b=u.lesson,x=u.concept,y=u.lab,j=u.labResult,g=(0,m.Z)().goTo,w=(0,l.$)().t,C=(0,p.z)().track,k=(0,i.useMemo)((function(){return{part_key:null===d||void 0===d?void 0:d.key,lesson_key:null===b||void 0===b?void 0:b.key,concept_key:null===x||void 0===x?void 0:x.key}}),[null===x||void 0===x?void 0:x.key,null===b||void 0===b?void 0:b.key,null===d||void 0===d?void 0:d.key]),S=(0,i.useState)(!1),O=S[0],I=S[1],R=(0,i.useCallback)((function(e){e.stopPropagation(),C("Classroom Nav Clicked",W({},k,{title:"resource"})),I(!O)}),[O,C,k]),A=(0,i.useCallback)((function(e){if(e){var n=e.offsetParent;if(e.offsetTop<n.scrollTop)n.scrollTo({left:0,top:e.offsetTop,behavior:"smooth"});else{var r=e.offsetTop+e.offsetHeight;r>n.scrollTop+n.offsetHeight&&n.scrollTo({left:0,top:r-n.offsetHeight,behavior:"smooth"})}}}),[]),z=(0,i.useCallback)((function(e){C("Learning Nav Clicked",W({},k,{cta_type:"text"})),g.content(e)}),[g,C,k]);return(0,o.jsxs)(s.Kq,{flex:1,spacing:0,h:"full",bg:"blue-darker",children:[(0,o.jsxs)(s.kC,{justify:"space-between",alignItems:"center",bg:"blue-dark",shadow:"shadow-1",zIndex:"low",borderTop:"1px solid 'slate-darker'",borderBottom:"1px solid 'slate-darker'",p:1,ps:4,children:[(0,o.jsx)(s.X6,{size:"h5",as:"h5",color:"white",children:w("common.concepts")}),(0,o.jsx)(a.hU,{"aria-label":w("lesson.closeSidePanel"),icon:(0,o.jsx)(f.e,{children:(0,o.jsx)(c.rbG,{color:"white",w:8,h:8})}),onClick:r,variant:"minimal-inverse",isRound:!0,p:2})]}),(0,o.jsx)(G,{activeContent:x||y,labResult:j,lesson:b,onConceptClick:z,onScroll:A}),!!(null===b||void 0===b||null===(n=b.resources)||void 0===n?void 0:n.length)&&(0,o.jsxs)(s.xu,{children:[(0,o.jsxs)(s.kC,{justify:"space-between",alignItems:"center",bg:"blue-dark",shadow:"shadow-1",zIndex:"low",borderTop:"1px solid 'slate-darker'",borderBottom:"1px solid 'slate-darker'",p:1,ps:4,onClick:R,children:[(0,o.jsx)(s.X6,{size:"h5",as:"h5",color:"white",children:w("lesson.resources")}),(0,o.jsx)(a.hU,{"aria-label":w(O?"lesson.closeResources":"lesson.openResources"),icon:O?(0,o.jsx)(c.VAA,{color:"white",w:8,h:8}):(0,o.jsx)(c.y8P,{color:"white",w:8,h:8}),onClick:R,variant:"minimal-inverse",isRound:!0,p:2})]}),O&&(0,o.jsx)(s.QI,{m:0,p:0,listStyleType:"none",overflow:"auto",children:null===b||void 0===b?void 0:b.resources.map((function(e){return e.name&&e.uri?(0,o.jsx)(s.HC,{p:2,children:(0,o.jsx)(v.S,{label:e.name,href:e.uri,ctaText:e.name,rightIcon:(0,o.jsx)(c._8t,{w:8,h:8,role:"img","aria-label":w("common.download")}),variant:"minimal-inverse",color:"white",w:"100%",_hover:{bgColor:"rgba(255, 255, 255, 0.16)",color:"white"},trackingEventName:"Resource Clicked",trackingOptions:{resource_name:e.name,resource_type:t.FILE}})},e.uri):null}))})]}),(0,o.jsx)(T,{})]})};function V(e,n,r){return n in e?Object.defineProperty(e,n,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[n]=r,e}function Y(e){for(var n=1;n<arguments.length;n++){var r=null!=arguments[n]?arguments[n]:{},t=Object.keys(r);"function"===typeof Object.getOwnPropertySymbols&&(t=t.concat(Object.getOwnPropertySymbols(r).filter((function(e){return Object.getOwnPropertyDescriptor(r,e).enumerable})))),t.forEach((function(n){V(e,n,r[n])}))}return e}var ee="20rem",ne=function(e){var n=e.children,r=(0,l.$)(),t=r.i18n,a=(r.t,(0,u.qY)({defaultIsOpen:!0})),c=a.isOpen,f=a.onOpen,v=a.onClose,p=(0,i.useState)(!1),h=p[0],m=p[1],b=(0,i.useState)(),x=b[0],y=b[1],j=(0,i.useCallback)((function(e){m((function(n){return void 0!==e&&!n})),y((function(n){return n===e?void 0:e}))}),[]);return(0,o.jsxs)(d.M.Provider,{value:{isExpanded:h,expandedAtom:x,toggleExpandedAtom:j},children:[(0,o.jsx)(re,{isOpen:c,onClick:f}),(0,o.jsxs)(s.kC,{bg:"white",h:"full",children:[(0,o.jsx)(s.xu,Y({},"rtl"===t.dir()?{right:c?0:"-20rem"}:{left:c?0:"-20rem"},{position:"fixed",w:ee,top:16,bottom:0,shadow:"shadow-2",zIndex:"low",transition:"left 0.2s ease-in",children:(0,o.jsx)(J,{onClose:v})})),(0,o.jsx)(s.xu,{m:"auto",w:"full",mt:0,ps:c?ee:0,transition:"padding 0.2s ease-in",children:n})]})]})},re=function(e){var n=e.isOpen,r=e.onClick,t=(0,l.$)(),i=t.i18n,u=t.t;return(0,o.jsx)(s.xu,{pos:"fixed",mt:4,bgColor:"white",shadow:"shadow-1",zIndex:"low",opacity:n?0:1,transition:"opacity 0.2s ease-in",sx:"rtl"===i.dir()?{right:0,borderTopLeftRadius:"circle",borderBottomLeftRadius:"circle"}:{left:0,borderTopRightRadius:"circle",borderBottomRightRadius:"circle"},children:(0,o.jsx)(a.hU,{"aria-label":u("lesson.openSidePanel"),icon:(0,o.jsx)(c.mmj,{color:"slate",w:8,h:8}),variant:"minimal-inverse",isRound:!0,p:2,onClick:r,onMouseDown:function(e){return e.preventDefault()}})})},te=ne},7484:function(e,n,r){r.d(n,{t:function(){return i}});var t=r(63598),o=r(69549),i=function(){var e,n,r=(0,t.r)(),i=r.nanodegree,l=r.paidCourse,u=r.part,s=(0,o.B)().selectEnrollmentForKey;i?(e=i.key,(null===u||void 0===u?void 0:u.cloudResourcesServiceId)?n=null===u||void 0===u?void 0:u.cloudResourcesServiceId:(null===i||void 0===i?void 0:i.cloudResourcesServiceId)&&(n=null===i||void 0===i?void 0:i.cloudResourcesServiceId)):(null===l||void 0===l?void 0:l.cloudResourcesServiceId)&&(e=l.key,n=null===l||void 0===l?void 0:l.cloudResourcesServiceId);var a=s(null!==e&&void 0!==e?e:"");return{enrollmentId:null===a||void 0===a?void 0:a.id,serviceId:n}}},86190:function(e,n,r){r.d(n,{Z:function(){return s}});var t=r(67294),o=r(11163),i=r(7886),l=r(60299),u=r(63598);function s(){var e,n=(0,t.useState)(),r=n[0],s=n[1],a=(0,o.useRouter)(),c=(0,u.r)(),d=c.nanodegree,f=c.part,v=c.paidCourse,p=c.freeCourse,h=c.lesson,m=c.concept,b=c.lab,x=(0,t.useMemo)((function(){return d&&d.parts?d.parts:[]}),[d]),y=(0,t.useMemo)((function(){return f?f.lessons:v?v.lessons:p?p.lessons:[]}),[f,v,p]),j=(0,t.useCallback)((function(e,n){return d&&f?l.F.nanodegree(d,f,e,n):v?l.F.paidCourse(v,e,n):p?l.F.freeCourse(p,e,n):void 0}),[d,f,v,p]),g=(0,t.useMemo)((function(){return Math.max(x.findIndex((function(e){return e===f})),0)}),[x,f]),w=(0,t.useMemo)((function(){return Math.max(null!==(e=y.findIndex((function(e){return e===h})))&&void 0!==e?e:0,0)}),[y,h]),C=(0,t.useMemo)((function(){var e=h?(0,i.kz)(h):[],n=e.findIndex((function(e){return e===m})),r=e.findIndex((function(e){return e===b}));return Math.max(n,r,0)}),[h,m,b]),k=(0,t.useCallback)((function(e,n){var r=j(null!==n&&void 0!==n?n:h,e);r&&a.push(r)}),[j,h,a]),S=(0,t.useMemo)((function(){var e=h?(0,i.kz)(h):[];return C>=e.length-1}),[h,C]),O=(0,t.useMemo)((function(){var e=h?(0,i.kz)(h):[];if(!S){var n=e[C+1];return function(){return k(n)}}if(w<y.length-1){var r,t=y[w+1];s(null!==(r=t.title)&&void 0!==r?r:void 0);var o=(0,i.kz)(t)[0];return function(){k(o,t)}}if(g<x.length-1){var u,c,f=x[g+1],v=f.lessons[0];s(null!==(c=null===v||void 0===v?void 0:v.title)&&void 0!==c?c:"");var p=null===v||void 0===v||null===(u=v.concepts)||void 0===u?void 0:u[0],m="";if(d&&(m=l.F.nanodegree(d,f,v,p)),m&&!f.isExtraCurricular)return function(){a.push(m)}}var b=j();return s(void 0),b?function(){a.push(b)}:void 0}),[j,k,S,d,a,C,w,y,x,g,h]),I=(0,t.useMemo)((function(){var e=h?(0,i.kz)(h):[];if(C>0){var n=e[C-1];return function(){return k(n)}}if(w>0){var r=y[w-1],t=(0,i.kz)(r),o=t[t.length-1];return function(){k(o,r)}}var l=j();return l?function(){a.push(l)}:void 0}),[j,k,a,C,w,y,h]),R=(0,t.useMemo)((function(){var e=[];return d&&d.title&&e.push({name:d.title,path:l.F.nanodegree(d)}),d&&f&&f.title&&e.push({name:f.title,path:l.F.nanodegree(d,f)}),v&&v.title&&e.push({name:v.title,path:l.F.paidCourse(v)}),p&&p.title&&e.push({name:p.title,path:l.F.freeCourse(p)}),e}),[d,f,v,p]);return{goTo:{content:k,next:O,previous:I},isLastContent:S,nextLessonTitle:r,breadcrumbs:R}}},1921:function(e,n,r){r.d(n,{n:function(){return c},L:function(){return f}});var t=r(34051),o=r.n(t),i=r(51179),l=r(92098),u=r(63686);function s(e,n,r,t,o,i,l){try{var u=e[i](l),s=u.value}catch(a){return void r(a)}u.done?n(s):Promise.resolve(s).then(t,o)}function a(e){return function(){var n=this,r=arguments;return new Promise((function(t,o){var i=e.apply(n,r);function l(e){s(i,t,o,l,u,"next",e)}function u(e){s(i,t,o,l,u,"throw",e)}l(void 0)}))}}function c(e,n){return d.apply(this,arguments)}function d(){return(d=a(o().mark((function e(n,r){return o().wrap((function(e){for(;;)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,(0,u.v_)("".concat((0,i.H)("/api/cloud-resources/v2"),"/services/").concat(n,"/launch"),{enrollment_id:r});case 3:return e.abrupt("return",e.sent);case 6:throw e.prev=6,e.t0=e.catch(0),e.t0;case 9:case"end":return e.stop()}}),e,null,[[0,6]])})))).apply(this,arguments)}function f(e){return(0,l.zO)().add(10,"minutes").isAfter(e.session_end)}}}]);