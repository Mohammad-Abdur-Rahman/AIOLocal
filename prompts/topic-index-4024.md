# Maths past-paper topic index (user prompt, verbatim)

Run against the merged PDF (e.g. test.pdf, 204 pages).

---

You are a precise mathematics exam-indexing assistant. I am providing a merged PDF containing compiled past examination papers from **2019 to 2026**.

Your task is to scan **every single page from Page 1 to the final page of the merged PDF** and classify every mathematics question (and any standalone sub-question if a single question spans or mixes topics, though primarily map each question/item to the merged PDF page number(s) where it appears) into the topical categories defined below.

### Output Format Rules (Strict)
1. Output **only** the topic lines in the exact format:
   `Topic Name: x, y, z`
   where `x, y, z` are the **1-based page numbers of the merged PDF** (not internal printed question-paper page numbers, unless specified otherwise).
2. List page numbers in **ascending numerical order** and **deduplicate** repeated page numbers within the same topic (e.g., `Quadratic & Simultaneous Equations: 4, 4, 19` -> `Quadratic & Simultaneous Equations: 4, 19`).
3. If a topic has no matching questions in the PDF, still include the line with `None` (e.g., `Matrices & Legacy 2019 Topics: None`) so I know the entire document was checked for that category.
4. Do **not** skip pages or stop early—process the **entire document** from start to finish. If a question spans across two pages, include both page numbers where the relevant math appears. If a multi-part question tests two distinct topics on different pages, list each page under its respective topic; if sub-parts sit on the same page, list that page under every topic tested on that page.

---

### Topic Categories & Boundaries

#### 1. Number: Arithmetic, Fractions, Decimals, Percentages & Ratio
* **Includes:** Types of numbers (natural, integers, primes, prime factorisation, HCF/LCM, rational/irrational, reciprocals), ordering/inequality signs, four operations, fractions/decimals/percentages (including recurring decimals, percentage change, reverse percentages, profit/loss, discount, deposit, simple/compound interest), ratio and proportion (simplifying ratios, sharing in a ratio, map scales, best value), rates (speed/distance/time, exchange rates, currency conversion, hourly pay, density, pressure, flow/fuel rates), time/clocks/timetables/time zones, rounding/significant figures/estimation, upper and lower bounds (limits of accuracy), and exponential growth/decay word problems.
* **Excludes:** Algebraic proportion with `∝` and equations like `y = kx^2` (put in *Algebra: Functions, Graphs & Proportion*); surds (put in *Surds & Indices*); speed-time/travel graphs (put in *Graphs & Kinematics*).

#### 2. Surds & Indices
* **Includes:** Powers and roots, laws/rules of indices with numerical or algebraic bases (positive, zero, negative, and fractional indices), standard form (`A × 10^n`) calculations, simplifying surds, and rationalising denominators.
* **Excludes:** Exponential functions/curve sketching (put in *Graphs & Kinematics*).

#### 3. Algebra: Expressions, Factorisation & Algebraic Fractions
* **Includes:** Substitution and evaluating expressions/formulas, collecting like terms, expanding single/double/triple brackets, factorisation (common factors, grouping, difference of two squares, quadratics, cubics with common `x`), completing the square on an expression (without solving `= 0`), simplifying/adding/subtracting/multiplying/dividing algebraic fractions, and changing the subject of a formula.
* **Excludes:** Solving equations/inequalities (put in *Equations & Inequalities*).

#### 4. Equations & Inequalities
* **Includes:** Constructing and solving linear equations in one unknown, fractional equations leading to linear or quadratic equations, simultaneous linear equations (and linear + non-linear pairs if present), solving quadratic equations (factorisation, completing the square, quadratic formula), inequalities on a number line, solving linear inequalities, and graphing / identifying regions of linear inequalities in two variables.
* **Excludes:** Pure expression simplification without `=` or inequality signs (put in *Algebra: Expressions, Factorisation & Algebraic Fractions*).

#### 5. Sequences, Functions & Algebraic Proportion
* **Includes:** Number sequences and nth term rules (linear, quadratic, cubic, exponential, or pattern/dot/matchstick sequences), direct and inverse proportion expressed algebraically (`y ∝ x`, `y ∝ 1/x`, `y ∝ x^2`, `y ∝ √x`, etc., finding constants/unknowns), function notation `f(x)`, domain/range, inverse functions `f^-1(x)`, and composite functions `gf(x)` / `fg(x)`.
* **Excludes:** Everyday ratio/recipe/map-scale/speed questions without algebraic proportion equations (put in *Number: Arithmetic, Fractions, Decimals, Percentages & Ratio*).

#### 6. Coordinate Geometry
* **Includes:** Cartesian coordinates `(x, y)` in 2D, plotting/reading straight-line graphs, gradient of a line or between two points, length of a line segment (distance between two points), midpoint of a line segment, equation of a straight line (`y = mx + c`, `ax + by = c`, horizontal/vertical lines), parallel lines, and perpendicular lines / perpendicular bisectors.
* **Excludes:** Non-linear curves (put in *Graphs & Kinematics*); 2D geometric vectors with `→` or column vectors for general geometry/translation unless explicitly tied to coordinate grid line equations (put in *Transformations & Vectors*).

#### 7. Graphs & Kinematics
* **Includes:** Travel graphs, conversion graphs, distance-time and speed-time graphs, acceleration/deceleration, distance as area under a speed-time graph, tables of values and plotting/sketching/recognising linear, quadratic, cubic, reciprocal, and exponential graphs, finding roots or intersections graphically, and estimating gradients/tangents to curves.
* **Excludes:** Straight-line coordinate geometry tasks like midpoint, length, or perpendicular bisectors (put in *Coordinate Geometry*).

#### 8. Geometry: Angles, Polygons, Constructions, Bearings & Symmetry
* **Includes:** Angle properties (straight line, around a point, vertically opposite, triangles, quadrilaterals, parallel lines—corresponding, alternate, co-interior), interior/exterior angles of regular and irregular polygons, geometric constructions with ruler and compasses, nets (when focused on geometric net structure/construction), scale drawings, three-figure bearings (`000°` to `360°`), line/rotational symmetry, and planes/axes of symmetry.
* **Excludes:** Circle angle/chord/tangent theorems (put in *Circle Theorems*); trigonometry/SOHCAHTOA/sine/cosine rules (put in *Trigonometry*); area/volume/surface area calculations (put in *Mensuration*).

#### 9. Circle Theorems
* **Includes:** Angle in a semicircle (`90°`), angle between tangent and radius (`90°`), angle at centre = twice angle at circumference, angles in the same segment, opposite angles of a cyclic quadrilateral, alternate segment theorem, equal chords equidistant from centre, perpendicular bisector of a chord through centre, and equal tangents from an external point.
* **Excludes:** Basic circumference, arc length, or sector area/volume without circle theorem angle/tangent/chord deductions (put in *Mensuration*).

#### 10. Mensuration (Perimeter, Area, Surface Area & Volume)
* **Includes:** Metric unit conversions (length, area, volume, capacity, mass), perimeter and area of 2D shapes (rectangles, triangles, parallelograms, trapezia, compound shapes), circles/arcs/sectors (circumference, area, arc length, sector area), and surface area and volume of 3D solids (cuboids, prisms, cylinders, cones, spheres, hemispheres, pyramids, frustums, compound solids).
* **Excludes:** Length/area/volume ratios of **similar** shapes/solids using scale factors (put in *Similarity & Congruency*); right-angled/non-right-angled triangle trigonometry (put in *Trigonometry*).

#### 11. Similarity & Congruency
* **Includes:** Proving or identifying similar/congruent triangles or shapes, scale factors in similarity, and using length-to-area (`k^2`) and length-to-volume (`k^3`) relationships of similar 2D shapes and 3D solids.
* **Excludes:** Non-similar area/perimeter/volume calculations (put in *Mensuration*).

#### 12. Trigonometry (2D & 3D)
* **Includes:** Pythagoras' theorem in 2D and 3D, trigonometric ratios (`sin`, `cos`, `tan`) in right-angled triangles, angles of elevation and depression, bearings combined with trigonometry, sine rule and cosine rule (including obtuse angles / ambiguous case), area of a triangle using `1/2 ab sin C`, and 3D trigonometry (including angle between a line and a plane).
* **Excludes:** Pure bearing questions solvable by simple angle rules/scale drawing without Pythagoras or trig (put in *Geometry: Angles, Polygons, Constructions, Bearings & Symmetry*).

#### 13. Transformations & Vectors
* **Includes:** Reflections, rotations, enlargements (including positive, fractional, and negative scale factors), translations, combinations of transformations, column vectors (`(x; y)`), vector addition/subtraction/scalar multiplication, magnitude of a vector (`|a|`), position vectors, expressing vectors in terms of base/coplanar vectors, and vector geometry proofs (collinearity, parallel vectors, ratios).
* **Excludes:** Coordinate line equations/gradients/midpoints without transformations or vectors (put in *Coordinate Geometry*).

#### 14. Probability
* **Includes:** Probability scale (`0` to `1`), probability of a single event and complement `1 - P(A)`, relative and expected frequency, fairness/bias, sample space diagrams, Venn diagrams used for probability (`P(A ∩ B)`, `P(A ∪ B)`), and tree diagrams (with or without replacement).
* **Excludes:** Pure set notation/Venn diagrams without probability calculations (put in *Sets & Logic/Venn Diagrams*); statistical averages/charts (put in *Statistics*).

#### 15. Statistics
* **Includes:** Tally/two-way tables, reading/interpreting/comparing statistical data, mean, median, mode, range, estimated mean and modal class for grouped data, bar charts, pie charts, pictograms, frequency distributions, scatter diagrams and lines of best fit, cumulative frequency curves/quartiles/interquartile range/percentiles, and histograms with frequency density.
* **Excludes:** Probability tree/sample space questions (put in *Probability*).

#### 16. Sets & Logic/Venn Diagrams
* **Includes:** Set language and notation (`∈`, `∉`, `⊂`, `⊈`, `∪`, `∩`, `A'`, `∅`, `ξ` / universal set, `n(A)`), set-builder notation, and Venn diagrams involving set membership, intersections, unions, and element counts (when not framed as combined probability).
* **Excludes:** Probability questions using `P(...)` or tree diagrams (put in *Probability*).

#### 17. Legacy / Out-of-Current-Syllabus (2019 & Earlier Papers)
* **Includes:** Any question or sub-part appearing in older papers (such as **2019** or earlier) that tests topics or formats outside the current 2025–2027 Syllabus D 4024 scope—for example: **matrices** (addition, subtraction, scalar multiplication, matrix multiplication, determinant/inverse, transformation matrices), **linear programming** regions/objective functions, **vectors/matrices/geometry/number/statistics sub-topics** that were removed or altered between the 2019 syllabus and the 2025–2027 syllabus.
* **Rule:** Always check if a 2019 question fits a current syllabus topic first; use this category specifically for items that are **out of the current 2025–2027 syllabus** so they aren't lost or force-fitted.

---

### Final Verification Before Responding
* Double-check that every page from Page 1 to the last page of the PDF has been inspected.
* Ensure the output contains **only** the `Topic Name: x, y, z` lines (no intro/outro chatter unless I ask for notes).
