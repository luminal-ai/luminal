; -------- SYMBOLIC ALGEBRA -------
(ruleset expr)
(datatype Expression
	(MNum i64)
	(MVar String)
	(MAdd Expression Expression)
	(MSub Expression Expression)
	(MMul Expression Expression)
	(MDiv Expression Expression)
	(MMod Expression Expression)
	(MMin Expression Expression)
	(MMax Expression Expression)
	(MAnd Expression Expression)
	(MOr Expression Expression)
	(MGte Expression Expression)
	(MLt Expression Expression)
	(MFloorTo Expression Expression)
    (MReplace Expression Expression Expression)
    (MAccum String) ; this marks that we feed the output (also marked with MAccum) back in
)

; Communative
(rewrite (MAdd a b) (MAdd b a) :ruleset expr)
(rewrite (MMul a b) (MMul b a) :ruleset expr)

; Associative
(rewrite (MAdd (MAdd a b) c) (MAdd a (MAdd b c)) :ruleset expr)
(rewrite (MMul (MMul a b) c) (MMul a (MMul b c)) :ruleset expr)

; Constant folding
(rewrite (MAdd (MNum a) (MNum b)) (MNum (+ a b)) :ruleset expr)
(rewrite (MSub (MNum a) (MNum b)) (MNum (- a b)) :ruleset expr)
(rewrite (MMul (MNum ?a) (MNum ?b)) (MNum (* ?a ?b)) :when ((< ?a 10000) (< ?b 10000)) :ruleset expr)
(rewrite (MDiv (MNum a) (MNum b)) (MNum (/ a b)) :when ((!= 0 b) (= 0 (% a b))) :ruleset expr)
(rewrite (MMax (MNum a) (MNum b)) (MNum (max a b)) :ruleset expr)
(rewrite (MMin (MNum a) (MNum b)) (MNum (min a b)) :ruleset expr)
(rewrite (MAnd (MNum a) (MNum b)) (MNum (& a b)) :ruleset expr)

; Simple reductions
(rewrite (MAdd a (MNum 0)) a :ruleset expr)
(rewrite (MMul a (MNum 1)) a :ruleset expr)
(rewrite (MMul a (MNum 0)) (MNum 0) :ruleset expr)
(rewrite (MDiv a (MNum 1)) a :ruleset expr)
(rewrite (MMod (MMul ?x ?y) ?y) (MNum 0) :ruleset expr)
(rewrite (MMod (MMod ?x (MNum ?y)) (MNum ?z)) (MMod ?x (MNum ?y)) :when ((>= ?z ?y) (= 0 (% ?y ?z))) :ruleset expr) ; nested mods
(rewrite (MMod (MMod ?x (MNum ?y)) (MNum ?z)) (MMod ?x (MNum ?z)) :when ((>= ?y ?z) (= 0 (% ?z ?y))) :ruleset expr)

; Replacement
(rewrite (MReplace ?x ?y ?z) ?z :when ((= ?x ?y)) :ruleset expr)
(rewrite (MReplace (MAdd ?a ?b) ?x ?y) (MAdd (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MSub ?a ?b) ?x ?y) (MSub (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMul ?a ?b) ?x ?y) (MMul (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MDiv ?a ?b) ?x ?y) (MDiv (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMod ?a ?b) ?x ?y) (MMod (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMin ?a ?b) ?x ?y) (MMin (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MMax ?a ?b) ?x ?y) (MMax (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
(rewrite (MReplace (MFloorTo ?a ?b) ?x ?y) (MFloorTo (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)
; leave numbers unchanged
(rewrite (MReplace (MNum ?n) ?x ?y) (MNum ?n) :ruleset expr)
(rewrite (MReplace (MAccum ?acc) ?x ?y) (MAccum ?acc) :ruleset expr)

; leave other vars unchanged
(rewrite (MReplace (MVar ?v) (MVar ?x) ?y) (MVar ?v) :when ((!= ?v ?x)) :ruleset expr)

(ruleset cleanup)
(rule ((= e (MReplace a b c))) ((delete (MReplace a b c))) :ruleset cleanup)

; -------- IR --------
(ruleset ir)
(ruleset ir-prop)
(datatype LoopType (Loop String Expression))
(datatype UnOp
	(Exp2)
	(Log2)
	(Sqrt)
	(Sin)
	(Recip)
	(Neg)
)
(datatype BinOp
	(Add)
	(Mul)
	(Max)
)
(datatype IR
	; General kernel stuff
   	(GMEM String)
   	(LoopIn IR LoopType Expression)
   	(LoopOut IR LoopType Expression)

    ; search helpers
    (Unary UnOp IR)
   	(Binary BinOp IR IR)

   	; propogation patterns
   	(SwapLoops IR String String) ; Swap two loops, identified by their strings
   	(TileLoop IR String) ; Tile a loop, identified by it's string
    (UnpadLoop IR String) ; Remove a padding loop, identified by it's string
    (MergeLoops IR String String) ; Merge loops, identified by their strings
    (Fused IR) ; Says that we have previously fused a loopout -> loopin here

   	; propogation pattern helpers
   	(PropOneArg String IR String) ; Generic prop one arg back
   	(PropTwoArgs String IR String String) ; Generic prop two args back

   	; tensor core stuff
   	(TCMatmul IR IR Expression Expression Expression Expression Expression Expression) ; input A, input B, A k stride, B k stride, A inner stride, B inner stride, C inner stride, number of K tile loops
   	(TiledMatmulInputA IR i64 Expression)
    (TiledMatmulInputB IR i64 Expression)
)

; -------------- HELPERS ---------------

; Communative binary ops
;(rewrite (Binary ?bin ?a ?b) (Binary ?bin ?b ?a) :ruleset ir)
; distributive/associative skeletons so sums and products re-associate
;(rewrite (Add (Add ?a ?b) ?c) (Add ?a (Add ?b ?c)) :ruleset ir)
;(rewrite (Mul (Mul ?a ?b) ?c) (Mul ?a (Mul ?b ?c)) :ruleset ir)

; set containing maccums
(sort ExpressionSetBase (Set Expression))

; a single global set, merged by union
(function MAccumSet () ExpressionSetBase :merge (set-union old new))

; for every (MAccum ...), add that exact term to the set
(rule
	((= ?e (MAccum ?s)))
	((set (MAccumSet) (set-of ?e)))
	:ruleset ir-prop
)

(function loop_level (IR) i64 :merge (max new old))
; GMEM (0) -> loopin (0)
(rule
	((= out (LoopIn (GMEM g) l1 r1)))
	(
		(set (loop_level out) 0)
		(set (loop_level (GMEM g)) 0)
	)
	:ruleset ir-prop
)
; non-loopin (n) -> loopout (n - 1)
(rule
	(
		(= curr (LoopOut x l1 r1))
		(!= x (LoopIn y l2 r2))
		(= xll (loop_level x))
	)
	((set (loop_level curr) (- xll 1)))
	:ruleset ir-prop
)
; loopin (n) -> binary (n + 1)
(rule
	(
		(= curr (Binary bin x z))
		(= x (LoopIn y l1 r1))
		(= xll (loop_level x))
	)
	((set (loop_level curr) (+ xll 1)))
	:ruleset ir-prop
)
; loopin (n) -> unary (n + 1)
(rule
	(
		(= curr (Unary un x))
		(= x (LoopIn y l1 r1))
		(= xll (loop_level x))
	)
	((set (loop_level curr) (+ xll 1)))
	:ruleset ir-prop
)
; loopin (n) -> loopin (n + 1)
(rule
	(
		(= curr (LoopIn x l2 r2))
		(= x (LoopIn y l1 r1))
		(= xll (loop_level x))
	)
	((set (loop_level curr) (+ xll 1)))
	:ruleset ir-prop
)
; loopin (n) -> loopout (n)
(rule
	(
		(= curr (LoopOut x l1 r1))
		(= x (LoopIn y l2 r2))
		(= xll (loop_level x))
	)
	((set (loop_level curr) xll))
	:ruleset ir-prop
)
; loopout (n) -> loopin (n)
(rule
	(
		(= curr (LoopIn x l1 r1))
		(= x (LoopOut y l2 r2))
		(= xll (loop_level x))
	)
	((set (loop_level curr) xll))
	:ruleset ir-prop
)
; non-loopin -> binary
(rule
	(
		(= curr (Binary bin a b))
		(!= a (LoopIn c l2 r2))
		(= xll (loop_level a))
	)
	((set (loop_level curr) xll))
	:ruleset ir-prop
)
; non-loopin -> unary
(rule
	(
		(= curr (Unary un a))
		(!= a (LoopIn c l2 r2))
		(= xll (loop_level a))
	)
	((set (loop_level curr) xll))
	:ruleset ir-prop
)
; loopin -> tcmatmul
(rule
	(
		(= curr (TCMatmul a b c d e f g h))
		(= a (LoopIn x l1 r1))
		(= xll (loop_level a))
	)
	((set (loop_level curr) (+ xll 1)))
	:ruleset ir-prop
)
; loopout -> loopout
(rule
	(
		(= curr (LoopOut x l1 r1))
		(= x (LoopOut y l2 r2))
		(= xll (loop_level x))
	)
	((set (loop_level curr) (- xll 1)))
	:ruleset ir-prop
)

; ---------- RULES ----------

; Loop Fusion
(rewrite
	(LoopIn (LoopOut (Binary ?bin ?a ?b) (Loop ?loopA ?range) ?st) (Loop ?loopB ?range) ?st)
	(Fused (Binary ?bin ?a ?b))
	:ruleset ir
)
(rewrite
	(LoopIn (LoopIn
		(LoopOut (LoopOut (Binary ?bin ?a ?b) (Loop ?loopA1 ?range1) ?st1) (Loop ?loopA2 ?range2) ?st2)
	(Loop ?loopB2 ?range2) ?st2) (Loop ?loopB1 ?range1) ?st1)
	(Fused (Binary ?bin ?a ?b))
	 :ruleset ir
)
(rewrite
	(LoopIn (LoopIn (LoopIn
		(LoopOut (LoopOut (LoopOut
			(Binary ?bin ?a ?b)
		(Loop ?loopA1 ?range1) ?st1) (Loop ?loopA2 ?range2) ?st2) (Loop ?loopA3 ?range3) ?st3)
	(Loop ?loopB3 ?range3) ?st3) (Loop ?loopB2 ?range2) ?st2) (Loop ?loopB1 ?range1) ?st1)
	(Fused (Binary ?bin ?a ?b))
	:ruleset ir
)
(rewrite
	(LoopIn (LoopOut (Unary ?un ?a) (Loop ?loopA ?range) ?st) (Loop ?loopB ?range) ?st)
	(Fused (Unary ?un ?a))
	:ruleset ir
)
(rewrite
	(LoopIn (LoopIn
		(LoopOut (LoopOut (Unary ?un ?a) (Loop ?loopA1 ?range1) ?st1) (Loop ?loopA2 ?range2) ?st2)
	(Loop ?loopB2 ?range2) ?st2) (Loop ?loopB1 ?range1) ?st1)
	(Fused (Unary ?un ?a))
	 :ruleset ir
)
(rewrite
	(LoopIn (LoopIn (LoopIn
		(LoopOut (LoopOut (LoopOut
			(Unary ?un ?a)
		(Loop ?loopA1 ?range1) ?st1) (Loop ?loopA2 ?range2) ?st2) (Loop ?loopA3 ?range3) ?st3)
	(Loop ?loopB3 ?range3) ?st3) (Loop ?loopB2 ?range2) ?st2) (Loop ?loopB1 ?range1) ?st1)
	(Fused (Unary ?un ?a))
	:ruleset ir
)

; Tiling
(rewrite
	(LoopOut ?body (Loop ?loop (MNum ?range)) ?stride)
	(LoopOut
		(LoopOut
			(TileLoop ?body ?loop)
			(Loop (+ ?loop "_tile") (MNum 8))
			?stride
		)
		(Loop (+ ?loop "_out") (MNum (/ ?range 8)))
		(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum 8)))
	)
	:when ((> ?range 8) (= (% ?range 8) 0))
	;:ruleset ir
)
(rewrite
	(TileLoop (LoopIn ?body (Loop ?loop (MNum ?range)) ?stride) ?loop)
	(LoopIn
		(LoopIn ?body
			(Loop (+ ?loop "_out") (MNum (/ ?range 8)))
			(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum 8)))
		)
		(Loop (+ ?loop "_tile") (MNum 8))
		?stride
	)
	:ruleset ir-prop
)
; propogation
(rewrite
	(TileLoop (LoopIn ?body (Loop ?other ?range) ?stride) ?loop)
	(LoopIn (TileLoop ?body ?loop) (Loop ?other ?range) ?stride)
	:when ((!= ?loop ?other))
	:ruleset ir-prop
)
(rewrite
	(TileLoop (LoopOut ?body (Loop ?other ?range) ?stride) ?loop)
	(LoopOut (TileLoop ?body ?loop) (Loop ?other ?range) ?stride)
	 :ruleset ir-prop
)
(rewrite
	(TileLoop (Unary ?un ?body) ?loop)
	(Unary ?un (TileLoop ?body ?loop))
	 :ruleset ir-prop
)
(rewrite
	(TileLoop (Binary ?bin ?bodyA ?bodyB) ?loop)
	(Binary ?bin (TileLoop ?bodyA ?loop) (TileLoop ?bodyB ?loop))
	 :ruleset ir-prop
)


; Loop merging
(rewrite
	(LoopOut
		(LoopOut ?x
			(Loop ?i (MNum ?rangeI)) ?stI
		)
		(Loop ?o (MNum ?rangeO)) ?stO
	)
	(LoopOut (MergeLoops ?x ?o ?i)
		(Loop (+ ?o (+ "merge" ?i)) (MNum (* ?rangeO ?rangeI)))
		(MAdd (MReplace ?stO (MVar "z") (MDiv (MVar "z") (MNum ?rangeI))) (MReplace ?stI (MVar "z") (MMod (MVar "z") (MNum ?rangeI))))
	)
	:when ((set-not-contains (MAccumSet) ?stI) (set-not-contains (MAccumSet) ?stO))
	:ruleset ir
)
(rewrite
	(MergeLoops
		(LoopIn
			(LoopIn
				?x
				(Loop ?o ?rangeO) ?stO
			)
			(Loop ?i ?rangeI) ?stI
		)
		?o ?i
	)
	(LoopIn
		?x
		(Loop (+ ?o (+ "merge" ?i)) (MMul ?rangeO ?rangeI))
		(MAdd (MReplace ?stO (MVar "z") (MDiv (MVar "z") ?rangeI)) (MReplace ?stI (MVar "z") (MMod (MVar "z") ?rangeI)))
	)
	 :ruleset ir-prop
)
; propogation
(rewrite
	(MergeLoops (LoopIn ?body (Loop ?other ?range) ?stride) ?o ?i)
	(LoopIn (MergeLoops ?body ?o ?i) (Loop ?other ?range) ?stride)
	:when ((!= ?i ?other))
	:ruleset ir-prop
)
(rewrite
	(MergeLoops (LoopOut ?body (Loop ?other ?range) ?stride) ?o ?i)
	(LoopOut (MergeLoops ?body ?o ?i) (Loop ?other ?range) ?stride)
	 :ruleset ir-prop
)
(rewrite
	(MergeLoops (Unary ?un ?body) ?o ?i)
	(Unary ?un (MergeLoops ?body ?o ?i))
	 :ruleset ir-prop
)
(rewrite
	(MergeLoops (Binary ?bin ?bodyA ?bodyB) ?o ?i)
	(Binary ?bin (MergeLoops ?bodyA ?o ?i) (MergeLoops ?bodyB ?o ?i))
	 :ruleset ir-prop
)

; TensorCore
(ruleset tc)
(rewrite
	(LoopIn ; k
		(LoopIn ; n
			(LoopIn ; m
				?a
				(Loop ?loop_m (MNum ?m))
				(MMul (MVar "z") (MNum ?k))
			)
			(Loop ?loop_n (MNum ?n))
			(MNum 0)
		)
		(Loop ?loop_k (MNum ?k))
		(MVar "z")
	)
	(TiledMatmulInputA ?a ?k (MNum (/ ?k 8)))
	;:when ((= (% ?k 8) 0) (= (% ?m 8) 0) (= (% ?n 8) 0))
	:ruleset tc
)
(rewrite
	(LoopIn ; k
		(LoopIn ; n
			(LoopIn ; m
				?b
				(Loop ?loop_m (MNum ?m))
				(MNum 0)
			)
			(Loop ?loop_n (MNum ?n))
			(MVar "z")
		)
		(Loop ?loop_k (MNum ?k))
		(MMul (MVar "z") (MNum ?n))
	)
	(TiledMatmulInputB ?b ?n (MNum (/ ?k 8)))
	;:when ((= (% ?k 8) 0) (= (% ?m 8) 0) (= (% ?n 8) 0))
	:ruleset tc
)
(rewrite
	(LoopOut ; m
		(LoopOut ; n
			 (LoopOut ; k
				(Binary (Add)
					(Fused (Binary (Mul)
						(TiledMatmulInputB ?b ?n ?k_loops)
						(TiledMatmulInputA ?a ?k ?k_loops)
					))
					; accumulator
					(LoopIn ; k
						(LoopIn ; n
							(LoopIn ; m
								?acc
								(Loop ?loop_acc_mtile (MNum ?m))
								(MNum 0)
							)
							(Loop ?loop_acc_ntile (MNum ?n))
							(MNum 0)
						)
						(Loop ?loop_acc_k (MNum ?k))
						(MAccum ?accum)
					)
				)
				(Loop ?loop_out_k (MNum ?k))
				(MAccum ?acc_outer)
			)
			(Loop ?loop_out_n (MNum ?n))
			(MVar "z")
		)
		(Loop ?loop_out_m (MNum ?m))
		(MMul (MVar "z") (MNum ?n))
	)
	(LoopOut ; m outer
		(LoopOut ; n outer
			(LoopOut ; m tile
				(LoopOut ; n tile
					(TCMatmul
						; a
						(LoopIn ; n tile
							(LoopIn ; m tile
								(LoopIn ; n outer
									(LoopIn ; m outer
										?a
										(Loop ?loop_out_m (MNum (/ ?m 8)))
										(MMul (MVar "z") (MNum (* ?k 8)))
									)
									(Loop ?loop_out_n (MNum (/ ?n 8)))
									(MNum 0)
								)
								(Loop (+ ?loop_out_m "_tile") (MNum 8))
								(MNum 0)
							)
							(Loop (+ ?loop_out_n "_tile") (MNum 4))  ; each thread in the matmul does 2 elements
							(MNum 0)
						)
						; b
						(LoopIn ; n tile
							(LoopIn ; m tile
								(LoopIn ; n outer
									(LoopIn ; m outer
										?b
										(Loop ?loop_out_m (MNum (/ ?m 8)))
										(MNum 0)
									)
									(Loop ?loop_out_n (MNum (/ ?n 8)))
									(MMul (MVar "z") (MNum 8))
								)
								(Loop (+ ?loop_out_m "_tile") (MNum 8))
								(MNum 0)
							)
							(Loop (+ ?loop_out_n "_tile") (MNum 4))  ; each thread in the matmul does 2 elements
							(MNum 0)
						)
						; a k stride
						(MMul (MVar "z") (MNum 8))
						; b k stride
						(MMul (MVar "z") (MNum (* ?n 8)))
						; a row size
						(MNum ?k)
						; b row size
						(MNum ?n)
						; c row size
						(MNum ?n)
						; k loops
						?k_loops
					)
					(Loop (+ ?loop_out_n "_tile") (MNum 4))
					(MNum 0)
				)
				(Loop (+ ?loop_out_m "_tile") (MNum 8))
				(MNum 0)
			)
			(Loop ?loop_out_n (MNum (/ ?n 8)))
			(MMul (MVar "z") (MNum 8))
		)
		(Loop ?loop_out_m (MNum (/ ?m 8)))
		(MMul (MVar "z") (MNum (* ?n 8)))
	)
	:ruleset tc
)

; Swap loops
(rewrite
	(LoopOut
		(LoopOut
			?x
			(Loop ?innerLoop ?innerRange)
			?innerStride
		)
		(Loop ?outerLoop ?outerRange)
		?outerStride
	)
	(LoopOut
		(LoopOut
			(SwapLoops
				?x
				?innerLoop
				?outerLoop
			)
			(Loop ?outerLoop ?outerRange)
			?outerStride
		)
		(Loop ?innerLoop ?innerRange)
		?innerStride
	)
	:when ((set-not-contains (MAccumSet) ?innerStride) (!= ?innerLoop ?outerLoop))
	;:ruleset ir
)
(rewrite
	(SwapLoops
		(LoopIn
			(LoopIn
				?x
				(Loop ?outerLoop ?outerRange)
				?outerStride
			)
			(Loop ?innerLoop ?innerRange)
			?innerStride
		)
		?innerLoop
		?outerLoop
	)
	(LoopIn
		(LoopIn
			?x
			(Loop ?innerLoop ?innerRange)
			?innerStride
		)
		(Loop ?outerLoop ?outerRange)
		?outerStride
	)
	:ruleset ir-prop
)
; propogate
(rewrite
	(SwapLoops (LoopOut ?x ?loop ?stride) ?innerLoop ?outerLoop)
	(LoopOut (SwapLoops ?x ?innerLoop ?outerLoop) ?loop ?stride)
	:ruleset ir-prop
)
(rewrite
	(SwapLoops (LoopIn ?x (Loop ?loop ?range) ?stride) ?innerLoop ?outerLoop)
	(LoopIn (SwapLoops ?x ?innerLoop ?outerLoop) (Loop ?loop ?range) ?stride)
	:when ((!= ?loop ?innerLoop))
	:ruleset ir-prop
)
(rewrite
	(SwapLoops (Unary ?un ?a) ?innerLoop ?outerLoop)
	(Unary ?un (SwapLoops ?a ?innerLoop ?outerLoop))
	:ruleset ir-prop
)
(rewrite
	(SwapLoops (Binary ?bin ?a ?b) ?innerLoop ?outerLoop)
	(Binary ?bin (SwapLoops ?a ?innerLoop ?outerLoop) (SwapLoops ?b ?innerLoop ?outerLoop))
	:ruleset ir-prop
)

{code}

(ruleset loop-unname)
(rewrite (Loop ?s ?r) (Loop "" ?r) :ruleset loop-unname)

(run-schedule
	(saturate expr)
	(let-scheduler bo (back-off))
	(repeat 1
		(run-with bo ir)
		(saturate ir-prop)
		(saturate expr)
		(saturate cleanup)
	)
	(saturate ir-prop)
	(saturate tc)
	(saturate loop-unname) ; TODO: we need to get rid of loop names entirely
)

;(print-size)