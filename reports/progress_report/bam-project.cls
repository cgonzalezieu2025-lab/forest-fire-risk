\NeedsTeXFormat{LaTeX2e}[2020/10/01]
\ProvidesClass{bam-project}[2026/07/23 v1.2 IE BAM general group-project template]

% ------------------------------------------------------------
% LOCAL CHANGES (forest-fire-risk progress report, Oct 2026)
% The IE Overleaf server has a minimal TeX Live without several
% packages this class needs. Changes, all marked "LOCAL":
%  - \BAMplainfonts (defined in main.tex) skips the T1 font
%    encoding, whose fonts the server lacks; text stays Computer Modern.
%  - algorithm, algpseudocode, listings, tikz and tcolorbox load only
%    if installed; the guidance boxes fall back to plain grey text.
% With a full TeX Live the class behaves exactly as the official one.
% ------------------------------------------------------------

% ============================================================
% CLASS OPTIONS
% Use one layout option and one guidance option:
%   onecolumn or twocolumn
%   annotated or clean
% ============================================================
\newif\ifBAMtwocolumn
\newif\ifannotated
\BAMtwocolumnfalse
\annotatedtrue

\DeclareOption{onecolumn}{\BAMtwocolumnfalse}
\DeclareOption{twocolumn}{\BAMtwocolumntrue}
\DeclareOption{annotated}{\annotatedtrue}
\DeclareOption{clean}{\annotatedfalse}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ExecuteOptions{onecolumn,annotated}
\ProcessOptions\relax

\ifBAMtwocolumn
  \LoadClass[10pt,a4paper,twocolumn]{article}
\else
  \LoadClass[11pt,a4paper,onecolumn]{article}
\fi

% ============================================================
% ENCODING, LANGUAGE, TYPOGRAPHY, AND PAGE LAYOUT
% ============================================================
\RequirePackage{iftex}
\ifPDFTeX
  \ifdefined\BAMplainfonts\else % LOCAL
    \RequirePackage[T1]{fontenc}
  \fi
\fi
\RequirePackage[english]{babel}
\ifBAMtwocolumn
  \RequirePackage[a4paper,left=18mm,right=18mm,top=20mm,bottom=22mm,headheight=14pt]{geometry}
\else
  \RequirePackage[a4paper,left=27mm,right=27mm,top=24mm,bottom=24mm,headheight=15pt]{geometry}
\fi
\RequirePackage{setspace}
\RequirePackage{microtype}
\RequirePackage{enumitem}

% ============================================================
% MATHEMATICS, TABLES, FIGURES, ALGORITHMS, AND CODE
% ============================================================
\RequirePackage{mathtools}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{array}
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{float}
% LOCAL: optional packages, loaded only when installed.
\newif\ifBAM@tcb
\IfFileExists{algorithm.sty}{\RequirePackage{algorithm}}{}
\IfFileExists{algpseudocode.sty}{\RequirePackage{algpseudocode}}{}
\IfFileExists{listings.sty}{\RequirePackage{listings}}{}
\RequirePackage[table]{xcolor}
\IfFileExists{tikz.sty}{%
  \RequirePackage{tikz}%
  \usetikzlibrary{arrows.meta,positioning,calc,intersections}%
}{}
\IfFileExists{tcolorbox.sty}{\RequirePackage[most]{tcolorbox}\BAM@tcbtrue}{}

% ============================================================
% STRUCTURE, HEADERS, AND LINKS
% ============================================================
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{xurl}
\RequirePackage[pdfencoding=auto,psdextra]{hyperref}
\RequirePackage{bookmark}

% ============================================================
% BAM VISUAL IDENTITY
% ============================================================
\definecolor{BAMblue}{HTML}{080886}
\definecolor{BAMgreen}{HTML}{70D700}
\definecolor{BAMlight}{HTML}{EEF3FA}
\definecolor{BAMborder}{HTML}{AAB8CE}
\definecolor{BAMgray}{HTML}{5F6670}
\definecolor{BAMwarning}{HTML}{B00020}

% ============================================================
% BODY TEXT AND PARAGRAPH SETTINGS
% ============================================================
\ifBAMtwocolumn
  \setstretch{1.02}
  \setlength{\parskip}{0.38em}
  \setlength{\columnsep}{7mm}
\else
  \setstretch{1.08}
  \setlength{\parskip}{0.52em}
\fi
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{2em}
\raggedbottom
\urlstyle{same}
\setlist{itemsep=0.25em,topsep=0.45em}

\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

% ============================================================
% HEADING HIERARCHY
% ============================================================
\ifBAMtwocolumn
  \titleformat{\section}
    {\normalfont\large\bfseries}
    {\thesection}{0.65em}{}
  \titleformat{\subsection}
    {\normalfont\normalsize\bfseries}
    {\thesubsection}{0.60em}{}
  \titleformat{\subsubsection}
    {\normalfont\normalsize\bfseries\itshape}
    {\thesubsubsection}{0.55em}{}
  \titlespacing*{\section}{0pt}{2.0ex plus 0.4ex minus 0.2ex}{0.8ex}
  \titlespacing*{\subsection}{0pt}{1.5ex plus 0.3ex minus 0.2ex}{0.5ex}
  \titlespacing*{\subsubsection}{0pt}{1.2ex plus 0.2ex minus 0.1ex}{0.4ex}
\else
  \titleformat{\section}
    {\normalfont\Large\bfseries}
    {\thesection}{0.75em}{}
  \titleformat{\subsection}
    {\normalfont\large\bfseries}
    {\thesubsection}{0.70em}{}
  \titleformat{\subsubsection}
    {\normalfont\normalsize\bfseries}
    {\thesubsubsection}{0.65em}{}
  \titlespacing*{\section}{0pt}{2.3ex plus 0.5ex minus 0.2ex}{1.0ex}
  \titlespacing*{\subsection}{0pt}{1.8ex plus 0.4ex minus 0.2ex}{0.7ex}
  \titlespacing*{\subsubsection}{0pt}{1.5ex plus 0.3ex minus 0.2ex}{0.5ex}
\fi
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}

% ============================================================
% CAPTIONS AND FLOATS
% ============================================================
\captionsetup{
  font=small,
  labelfont=bf,
  labelsep=period,
  justification=justified,
  singlelinecheck=false,
  skip=5pt
}
\captionsetup[table]{position=top}
\captionsetup[figure]{position=bottom}
\@ifpackageloaded{algorithm}{\captionsetup[algorithm]{position=top}}{} % LOCAL

\setlength{\textfloatsep}{14pt plus 2pt minus 3pt}
\setlength{\floatsep}{11pt plus 2pt minus 2pt}
\setlength{\intextsep}{11pt plus 2pt minus 2pt}
\renewcommand{\topfraction}{0.88}
\renewcommand{\bottomfraction}{0.75}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.75}
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{3}
\setcounter{totalnumber}{6}

\graphicspath{{./}{figures/}{images/}}
\DeclareGraphicsExtensions{.pdf,.png,.jpg,.jpeg,.eps}

% ============================================================
% AUTOMATIC SEMESTER AND ACADEMIC YEAR
% January--May: Spring of previous--current academic year.
% September--December: Fall of current--next academic year.
% June--August defaults to the upcoming Fall term; both values
% can be overridden in main.tex when preparing an earlier report.
% ============================================================
\newcount\BAM@academicyearstart
\newcount\BAM@academicyearend
\ifnum\month<6
  \gdef\BAM@semester{Spring}
  \BAM@academicyearstart=\numexpr\year-1\relax
  \BAM@academicyearend=\year
\else
  \gdef\BAM@semester{Fall}
  \BAM@academicyearstart=\year
  \BAM@academicyearend=\numexpr\year+1\relax
\fi
\edef\BAM@academicyear{\number\BAM@academicyearstart--\number\BAM@academicyearend}

% ============================================================
% PROJECT METADATA
% ============================================================
\newcommand{\BAM@title}{[Insert project title]}
\newcommand{\BAM@shorttitle}{[Short project title]}
\newcommand{\BAM@course}{[Insert course name or collaborating courses]}
\newcommand{\BAM@team}{[Team member 1]\\[0.2em][Team member 2]\\[0.2em][Team member 3]}
\newcommand{\BAM@instructors}{[Professor name or names]}
\newcommand{\BAM@submissiondate}{\today}
\newcommand{\BAM@degree}{Bachelor in Applied Mathematics}
\newcommand{\BAM@institution}{IE University}
\newcommand{\BAM@school}{School of Science \& Technology}
\newcommand{\BAM@logo}{ie-sst-logo.jpeg}
\newcommand{\BAM@bibstyle}{amsplain}
\newcommand{\BAM@bibfile}{references}

\newcommand{\BAMtitle}[1]{\gdef\BAM@title{#1}}
\newcommand{\BAMshorttitle}[1]{\gdef\BAM@shorttitle{#1}}
\newcommand{\BAMcourse}[1]{\gdef\BAM@course{#1}}
\newcommand{\BAMteam}[1]{\gdef\BAM@team{#1}}
\newcommand{\BAMinstructors}[1]{\gdef\BAM@instructors{#1}}
\newcommand{\BAMsemester}[1]{\gdef\BAM@semester{#1}}
\newcommand{\BAMacademicyear}[1]{\gdef\BAM@academicyear{#1}}
\newcommand{\BAMsubmissiondate}[1]{\gdef\BAM@submissiondate{#1}}
\newcommand{\BAMdegree}[1]{\gdef\BAM@degree{#1}}
\newcommand{\BAMinstitution}[1]{\gdef\BAM@institution{#1}}
\newcommand{\BAMschool}[1]{\gdef\BAM@school{#1}}
\newcommand{\BAMlogo}[1]{\gdef\BAM@logo{#1}}
\newcommand{\BAMbibliographystyle}[1]{\gdef\BAM@bibstyle{#1}}
\newcommand{\BAMbibliographyfile}[1]{\gdef\BAM@bibfile{#1}}

% ============================================================
% ANNOTATIONS AND PLACEHOLDERS
% ============================================================
\newcommand{\BAMattention}[1]{{\color{BAMwarning}#1}}

\ifBAM@tcb % LOCAL
\newtcolorbox{BAMannotationbox}{
  enhanced,
  breakable,
  colback=BAMlight,
  colframe=BAMborder,
  boxrule=0.55pt,
  arc=1.5pt,
  left=6pt,right=6pt,top=5pt,bottom=5pt,
  before skip=5pt,after skip=7pt,
  fontupper=\small
}
\else
\newenvironment{BAMannotationbox}%
  {\par\smallskip\begingroup\small\color{BAMgray}\noindent\ignorespaces}%
  {\par\endgroup\smallskip}
\fi


\newcommand{\annotation}[1]{%
  \ifannotated
    \begin{BAMannotationbox}#1\end{BAMannotationbox}%
  \fi
}

\newcommand{\placeholder}[1]{%
  \ifannotated
    \textit{\color{BAMgray}#1}%
  \fi
}

\newcommand{\BAMannotationnotice}{%
  \ifannotated
    \ifBAM@tcb % LOCAL
    \begin{tcolorbox}[
      enhanced,
      colback=BAMlight,
      colframe=BAMgreen,
      boxrule=0.8pt,
      arc=1.5pt,
      left=7pt,right=7pt,top=6pt,bottom=6pt,
      before skip=5pt,after skip=10pt
    ]
      \centering\small\bfseries
      ANNOTATED GUIDANCE COPY. Change the class option from
      \texttt{annotated} to \texttt{clean} before preparing the final report.
    \end{tcolorbox}
    \else
      \begin{center}\fcolorbox{BAMgreen}{BAMlight}{\parbox{0.9\linewidth}{\centering\small\bfseries
      ANNOTATED GUIDANCE COPY. Change the class option from
      \texttt{annotated} to \texttt{clean} before preparing the final report.}}\end{center}
    \fi
  \fi
}

% ============================================================
% HEADERS AND FOOTERS
% ============================================================
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\BAM@shorttitle}
\fancyhead[R]{\small Bachelor in Applied Mathematics}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

\fancypagestyle{plain}{%
  \fancyhf{}
  \fancyhead[L]{\small\BAM@shorttitle}
  \fancyhead[R]{\small Bachelor in Applied Mathematics}
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
  \renewcommand{\footrulewidth}{0pt}
}

% ============================================================
% TITLE PAGE
% The logo is intentionally small to keep the cover restrained.
% ============================================================
\newcommand{\makebamtitle}{%
  \pagenumbering{gobble}%
  \begin{titlepage}
    \thispagestyle{empty}
    \begingroup
      \setlength{\parindent}{0pt}
      \setlength{\parskip}{0pt}

      \begin{flushright}
        \IfFileExists{\BAM@logo}{%
          \includegraphics[width=2.75cm]{\BAM@logo}%
        }{%
          {\small\bfseries IE UNIVERSITY\\SCHOOL OF SCIENCE \& TECHNOLOGY}%
        }
      \end{flushright}

      \vspace{1.00cm}
      \begin{center}
        {\Large\bfseries Project Report\par}
        \vspace{1.1cm}
        \parbox{0.88\textwidth}{%
          \centering\LARGE\bfseries\BAM@title\par
        }

        \vspace{1.25cm}
        {\large\BAM@course\par}
        \vspace{0.35cm}
        {\large\BAM@degree\par}
        {\large\BAM@institution\par}

        \vfill

        {\normalsize\bfseries Team members\par}
        \vspace{0.35cm}
        {\large\BAM@team\par}

        \vfill

        {\normalsize\bfseries Professor(s)\par}
        \vspace{0.25cm}
        {\large\BAM@instructors\par}

        \vfill

        {\large\BAM@semester\ \BAM@academicyear\par}
        \vspace{0.35cm}
        {\normalsize Submission date: \BAM@submissiondate\par}
        \vspace*{0.25cm}
      \end{center}
    \endgroup
  \end{titlepage}
  \pagenumbering{arabic}%
  \setcounter{page}{1}%
}

% ============================================================
% DOCUMENT HELPERS
% ============================================================
\newenvironment{BAMabstract}{%
  \section*{Abstract}
  \phantomsection
  \addcontentsline{toc}{section}{Abstract}
}{\par}

\newcommand{\BAMkeywords}[1]{%
  \par\smallskip
  \noindent\textbf{Keywords:} #1\par
}

\newcommand{\BAMtableofcontents}{%
  \ifBAMtwocolumn\onecolumn\fi
  \tableofcontents
  \clearpage
  \ifBAMtwocolumn\twocolumn\fi
}

\newcommand{\BAMunnumberedsection}[1]{%
 % \clearpage
  \section*{#1}
  \phantomsection
  \addcontentsline{toc}{section}{#1}
  \markboth{#1}{#1}
}

\newcommand{\BAMappendices}{%
  \clearpage
  \appendix
}

\newcommand{\BAMprintbibliography}{%
 % \clearpage
  \IfFileExists{\BAM@bibfile.bib}{}{%
    \ClassWarningNoLine{bam-project}{Bibliography file `\BAM@bibfile.bib' was not found}%
  }%
  \phantomsection
  \addcontentsline{toc}{section}{\refname}
  \bibliographystyle{\BAM@bibstyle}
  \bibliography{\BAM@bibfile}
}

% ============================================================
% MATHEMATICAL ENVIRONMENTS
% ============================================================
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{assumption}[theorem]{Assumption}

\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

% ============================================================
% CODE LISTINGS
% ============================================================
\@ifpackageloaded{listings}{% LOCAL
\lstset{
  basicstyle=\ttfamily\small,
  numbers=left,
  numberstyle=\scriptsize\color{BAMgray},
  stepnumber=1,
  numbersep=7pt,
  frame=single,
  breaklines=true,
  breakatwhitespace=false,
  showstringspaces=false,
  columns=fullflexible,
  captionpos=b,
  keepspaces=true,
  upquote=true,
  aboveskip=0.9em,
  belowskip=0.9em
}
\renewcommand{\lstlistingname}{Listing}
\renewcommand{\lstlistlistingname}{List of Listings}
}{}

% ============================================================
% STANDARD NAMES AND PDF METADATA
% ============================================================
\renewcommand{\refname}{References}
\renewcommand{\contentsname}{Table of Contents}
\renewcommand{\listfigurename}{List of Figures}
\renewcommand{\listtablename}{List of Tables}

\AtBeginDocument{%
  \hypersetup{
    pdftitle={\BAM@title},
    pdfauthor={BAM Project Team},
    pdfsubject={Group project report for the Bachelor in Applied Mathematics at IE University},
    pdfcreator={LaTeX with the bam-project class}
  }%
  \ClassInfo{bam-project}{Semester set to \BAM@semester}
  \ClassInfo{bam-project}{Academic year set to \BAM@academicyear}
}

\endinput
