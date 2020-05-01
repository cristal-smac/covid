\NeedsTeXFormat{LaTeX2e}%
%
\ProvidesClass{cours-bdd}%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\LoadClass{beamer}
% Gestion des langues (polices, encodages, traduction)
\RequirePackage[francais]{babel}
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\RequirePackage{times}
% symboles
\RequirePackage[official,right]{eurosym}
\RequirePackage{amssymb}
% mise en forme
\RequirePackage[parfill]{parskip}
\RequirePackage{url}
%\RequirePackage[colorlinks=true, urlcolor=blue]{hyperref}
\RequirePackage{color}
\RequirePackage{graphicx}
\RequirePackage{verbatim}
\RequirePackage{moreverb} % pour verbatiminput  de programmes
\RequirePackage{multicol}
% dessin 
\RequirePackage{tikz}
\usetikzlibrary{positioning,shapes,calc,trees, fit}
% Création de MCD et Algèbre relationnelle
\RequirePackage{mcd}
\RequirePackage{algrela}
\RequirePackage[cachedir=_minted]{minted}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% utilisation du theme
%\usetheme{Boadilla}
\usetheme{ULille}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% zone de code sql
\newminted[sql]{sql}{keywordcase=upper}
\newmintinline{sql}{keywordcase=upper} % called using \sqlinline
\newmintedfile[sqlfile]{sql}{bgcolor=gray!20,keywordcase=upper}
% zone de code java
\newminted[java]{java}{bgcolor=gray!20}
\newmintinline{java}{bgcolor=gray!20} % called using \javainline
\newmintedfile[javafile]{java}{bgcolor=gray!20}
% zone de code html
\newminted[html]{html}{bgcolor=gray!20}
\newmintinline{html}{bgcolor=gray!20} % called using \htmlinline
\newmintedfile[htmlfile]{html}{bgcolor=gray!20}
% zone de code jsp
\newminted[jsp]{jsp}{bgcolor=gray!20}
\newmintinline{jsp}{bgcolor=gray!20} % called using \jspinline
\newmintedfile[jspfile]{jsp}{bgcolor=gray!20}
% zone de code javascript
\newminted[js]{js}{bgcolor=gray!20}
\newmintinline{js}{bgcolor=gray!20} % called using \jsinline
\newmintedfile[jsfile]{js}{bgcolor=gray!20,autogobble, tabsize=4}
% zone de code xml
\newminted[xml]{xml}{bgcolor=gray!20}
\newmintinline{xml}{bgcolor=gray!20} % called using \xmlinline
\newmintedfile[xmlfile]{xml}{bgcolor=gray!20,autogobble, tabsize=4} 
% commentaire d'un ordre SQL
\newenvironment{csql}{\medskip \noindent \em} {\vspace{-0.1cm}}

% ---------------------------------------------------------------------
% ------------- pour changer de section -------------------------------  

\newcommand{\changesection}[1]{
  \section{#1}
  \frame{\tableofcontents[currentsection,hideothersubsections]}
}

% ---------------------------------------------------------------------
% ------------------ commande de titre --------------------------------

\newcommand{\cristal}{
  \author{P.Mathieu}
  \institute[CRISTAL ULille]{SMAC Team
    Lille\\\href{https://www.cristal.univ-lille.fr/gt/i2c/}{https://www.cristal.univ-lille.fr/gt/i2c/}\\prenom.nom@univ-lille.fr}
}

\newcommand{\coursbdd}{
  \author{P.Mathieu}
  \institute[IUT-A Lille1]{IUT-A
    Lille\\\href{http://www.iut-a.univ-lille.fr}{http://www.iut-a.univ-lille.fr}\\prenom.nom@univ-lille.fr}
} 

\newcommand{\moocweb}{
  \author{Philippe Mathieu \& Guillaume Dufrene}
  \institute[IUT-A Lille1]{IUT-A
    Lille\\\href{http://www.iut-a.univ-lille.fr}{http://www.iut-a.univ-lille.fr}\\prenom.nom@univ-lille.fr}
} 

\newcommand{\courslp}{
  \author{P.Mathieu}
  \institute[LP DA2I Lille1]{LP DA2I
    Lille\\\href{http://www.iut-a.univ-lille.fr}{http://www.iut-a.univ-lille.fr}\\prenom.nom@univ-lille.fr}
} 

% ---------------------------------------------------------------------
% ------------------ macros -------------------------------------------

\newcommand{\entite}[1]{\texttt{#1}}
\newcommand{\asso}[1]{\textit{#1}}
\newcommand{\ul}[1]{\underline{#1}}
