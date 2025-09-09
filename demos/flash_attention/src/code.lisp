(datatype Math
	(MNum i64)
	(MVar String)
	(MAdd Math Math)
	(MSub Math Math)
	(MMul Math Math)
	(MDiv Math Math)
	(MMod Math Math)
	(MMin Math Math)
	(MMax Math Math)
	(MAnd Math Math)
	(MOr Math Math)
	(MGte Math Math)
	(MLt Math Math)
	(MFloorTo Math Math)
    (MReplace Math Math Math)
    (MAccum) ; this marks that we feed the output (also marked with MAccum) back in
)

; Communative
;(rewrite (MAdd a b) (MAdd b a))
;(rewrite (MMul a b) (MMul b a))
;(rewrite (MMin a b) (MMin b a))
;(rewrite (MMax a b) (MMax b a))
;(rewrite (MAnd a b) (MAnd b a))
;(rewrite (MOr a b) (MOr b a))

; Associative
(rewrite (MAdd (MAdd a b) c) (MAdd a (MAdd b c)))
(rewrite (MMul (MMul a b) c) (MMul a (MMul b c)))
;(rewrite (MDiv (MDiv a b) c) (MDiv a (MMul b c)))
;(rewrite (MDiv (MMul a b) c) (MMul a (MDiv b c)))
;(rewrite (MMul a (MDiv b c)) (MDiv (MMul a b) c))

; Distributive
;(rewrite (MMul a (MAdd b c)) (MAdd (MMul a b) (MMul a c)))
;(rewrite (MDiv (MAdd a b) c) (MAdd (MDiv a c) (MDiv b c)))

; Constant folding
(rewrite (MAdd (MNum a) (MNum b)) (MNum (+ a b)))
(rewrite (MSub (MNum a) (MNum b)) (MNum (- a b)))
(rewrite (MMul (MNum a) (MNum b)) (MNum (* a b)) :when ((< a 10000) (< b 10000)))
(rewrite (MDiv (MNum a) (MNum b)) (MNum (/ a b)) :when ((!= 0 b) (= 0 (% a b))))
(rewrite (MMax (MNum a) (MNum b)) (MNum (max a b)))
(rewrite (MMin (MNum a) (MNum b)) (MNum (min a b)))
(rewrite (MAnd (MNum a) (MNum b)) (MNum (& a b)))
;(rewrite (MOr (MNum a) (MNum b)) (MNum (| a b)))

; Factoring
;(rewrite (MAdd (MMul a b) (MMul a c)) (MMul a (MAdd b c)))
;(rewrite (MAdd a a) (MMul (MNum 2) a))

; Simple reductions
(rewrite (MAdd a (MNum 0)) a)
(rewrite (MMul a (MNum 1)) a)
(rewrite (MMul a (MNum 0)) (MNum 0))
(rewrite (MDiv a (MNum 1)) a)
(rewrite (MMul (MDiv ?a ?b) ?b) (MFloorTo ?a ?b))
(rewrite (MAdd (MFloorTo ?a ?b) (MMod ?a ?b)) ?a)


; Other
;(rewrite (MAdd (MDiv a b) c) (MDiv (MAdd a (MMul c b)) b))

; Replacement
(rewrite (MReplace ?x ?y ?z) ?z :when ((= ?x ?y)))
(rewrite (MReplace (MAdd ?a ?b) ?x ?y) (MAdd (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MSub ?a ?b) ?x ?y) (MSub (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MMul ?a ?b) ?x ?y) (MMul (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MDiv ?a ?b) ?x ?y) (MDiv (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MMod ?a ?b) ?x ?y) (MMod (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MMin ?a ?b) ?x ?y) (MMin (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MMax ?a ?b) ?x ?y) (MMax (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
(rewrite (MReplace (MFloorTo ?a ?b) ?x ?y) (MFloorTo (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))
;; leave numbers unchanged
(rewrite (MReplace (MNum ?n) ?x ?y) (MNum ?n))
(rewrite (MReplace (MAccum) ?x ?y) (MAccum))

;; leave other vars unchanged
(rewrite (MReplace (MVar ?v) (MVar ?x) ?y) (MVar ?v) :when ((!= ?v ?x)))


(datatype LoopType (Loop String Math))
(datatype*
 	(Expr
     	(Tensor String)
     	(LoopIn Expr LoopType Math)
     	(LoopOut Expr LoopType Math)
     	(Exp Expr)
     	(Sin Expr)
      	(Recip Expr)
       	(Neg Expr)
     	(Unary String Expr)
     	(Add Expr Expr)
     	(Mul Expr Expr)
      	(Max Expr Expr)
     	(Binary String Expr Expr)
     	(NewAcc i64) ; define accumulator for a loop
     	(AccOut Expr LoopType) ; retrieve the final result accumulator after the loop is complete
      	(SwapLoops Expr String String) ; Swap two loops, identified by their string
     )
)

; Convert to and from generic unary ops
(rewrite (Exp ?x) (Unary "Exp" ?x))
(rewrite (Unary "Exp" ?x) (Exp ?x))
(rewrite (Sin ?x) (Unary "Sin" ?x))
(rewrite (Unary "Sin" ?x) (Sin ?x))
(rewrite (Recip ?x) (Unary "Recip" ?x))
(rewrite (Unary "Recip" ?x) (Recip ?x))
(rewrite (Neg ?x) (Unary "Neg" ?x))
(rewrite (Unary "Neg" ?x) (Neg ?x))
(rewrite (Add ?a ?b) (Binary "Add" ?a ?b))
(rewrite (Binary "Add" ?a ?b) (Add ?a ?b))
(rewrite (Mul ?a ?b) (Binary "Mul" ?a ?b))
(rewrite (Binary "Mul" ?a ?b) (Mul ?a ?b))
(rewrite (Max ?a ?b) (Binary "Max" ?a ?b))
(rewrite (Binary "Max" ?a ?b) (Max ?a ?b))

; Communative binary ops
(rewrite (Binary ?bin ?a ?b) (Binary ?bin ?b ?a))
; distributive/associative skeletons so sums and products re-associate
(rewrite (Add (Add ?a ?b) ?c) (Add ?a (Add ?b ?c)))
(rewrite (Mul (Mul ?a ?b) ?c) (Mul ?a (Mul ?b ?c)))

; remove 1-level loop
(rewrite
 	(LoopOut (Unary ?un (LoopIn ?x (Loop ?loop (MNum 1)) (MVar "z"))) (Loop ?loop (MNum 1)) (MVar "z"))
	(Unary ?un ?x)
)
(rewrite
 	(LoopOut (Binary ?bin (LoopIn ?a (Loop ?loop (MNum 1)) (MVar "z")) (LoopIn ?b (Loop ?loop (MNum 1)) (MVar "z"))) (Loop ?loop (MNum 1)) (MVar "z"))
	(Binary ?bin ?a ?b)
)

; Loop Fusion
(rewrite (LoopIn (LoopOut ?x ?loop ?st) ?loop ?st) ?x)

; Loop Fission
(rewrite
	(LoopOut (Unary ?second (Unary ?first (LoopIn ?x ?loop ?inSt))) ?loop ?outSt)
	(LoopOut (Unary ?second (LoopIn (LoopOut (Unary ?first (LoopIn ?x ?loop ?inSt)) ?loop  (MVar "z")) ?loop (MVar "z"))) ?loop ?outSt)
)
(rewrite
	(Unary ?second (Binary ?first (LoopIn ?a ?loop ?inASt) (LoopIn ?b ?loop ?inBSt)))
	(Unary ?second (LoopIn (LoopOut (Binary ?first (LoopIn ?a ?loop ?inASt) (LoopIn ?b ?loop ?inBSt)) ?loop  (MVar "z")) ?loop (MVar "z")))
)
(rewrite
	(Binary ?second (Unary ?first (LoopIn ?a ?loop ?inASt)) (LoopIn ?b ?loop ?inBSt))
	(Binary ?second (LoopIn (LoopOut (Unary ?first (LoopIn ?a ?loop ?inASt)) ?loop  (MVar "z")) ?loop (MVar "z")) (LoopIn ?b ?loop ?inBSt))
)
(rewrite
	(Binary ?second (Binary ?first (LoopIn ?a ?loop ?inASt) (LoopIn ?c ?loop ?inCSt)) (LoopIn ?b ?loop ?inBSt))
	(Binary ?second (LoopIn (LoopOut (Binary ?first (LoopIn ?a ?loop ?inASt) (LoopIn ?c ?loop ?inCSt)) ?loop  (MVar "z")) ?loop (MVar "z")) (LoopIn ?b ?loop ?inBSt))
)
(rewrite (LoopOut (LoopOut
	(Unary
		?unHere
		(LoopIn (LoopOut ?inpA ?first ?firstStA) ?n ?inASt)
	) ?n ?st) ?lower ?lowerSt)
	(LoopOut (LoopOut
		(Unary
			?unHere
	       	(LoopIn	(LoopIn
	           	(LoopOut (LoopOut ?inpA ?first ?firstStA) ?lower ?lowerSt)
	       	?lower ?lowerSt) ?n ?inASt)
		)
	?n ?st) ?lower ?lowerSt)
)
(rewrite (LoopOut (LoopOut
    (Binary ?binHereA
    	(LoopIn (LoopOut ?inpB ?first ?firstStB) ?n ?inBSt)
        ?a
    ) ?n  ?st) ?lower ?lowerSt)
  	(LoopOut (LoopOut
        (Binary ?binHereA
        	(LoopIn (LoopIn
                (LoopOut (LoopOut ?inpB ?first ?firstStB) ?lower ?lowerSt)
            ?lower ?lowerSt) ?n ?inBSt)
            ?a
        )
    ?n  ?st) ?lower ?lowerSt)
)

; Loop merging
(rewrite
 	(LoopOut (LoopOut
       	(Unary ?merge
      		(LoopIn (LoopIn ?here (Loop ?outerL ?outer) ?outerStride) (Loop ?innerL ?inner) ?innerStride)
        )
    (Loop ?innerL ?inner) ?innerStride) (Loop ?outerL ?outer) ?outerStride)
 	(LoopOut
     	(Unary ?merge
           	(LoopIn ?here
               	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
               	(MAdd (MReplace ?outerStride (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStride (MVar "z") (MMod (MVar "z") ?inner)))
            )
    	)
     	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
		(MAdd (MReplace ?outerStride (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStride (MVar "z") (MMod (MVar "z") ?inner)))
    )
)
(rewrite
 	(LoopOut (LoopOut
       	(Binary
           	?binmerge
           	(LoopIn (LoopIn ?a (Loop ?outerL ?outer) ?outerStrideA) (Loop ?innerL ?inner) ?innerStrideA)
           	(LoopIn (LoopIn ?b (Loop ?outerL ?outer) ?outerStrideB) (Loop ?innerL ?inner) ?innerStrideB)
        )
    (Loop ?innerL ?inner) ?innerStride) (Loop  ?outerL ?outer) ?outerStride)
 	(LoopOut
     	(Binary ?binmerge
        	(LoopIn
             	?a
               	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
               	(MAdd (MReplace ?outerStrideA (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStrideA (MVar "z") (MMod (MVar "z") ?inner)))
            )
            (LoopIn
             	?b
               	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
               	(MAdd (MReplace ?outerStrideB (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStrideB (MVar "z") (MMod (MVar "z") ?inner)))
            )
    	)
     	(Loop (+ ?outerL ?innerL) (MMul ?inner ?outer))
		(MAdd (MReplace ?outerStride (MVar "z") (MDiv (MVar "z") ?inner)) (MReplace ?innerStride (MVar "z") (MMod (MVar "z") ?inner)))
    )
)
; Split loops
(let splitFactor 8)
(rewrite
 	(LoopOut (Unary ?spun (LoopIn ?x (Loop ?loopL (MNum ?loop)) ?stride)) (Loop ?loopL (MNum ?loop)) ?stride)
 	(LoopOut
     	(LoopOut
         	(Unary ?spun
            	(LoopIn
                 	(LoopIn
                     	?x
                     	(Loop ?loopL (MNum (/ ?loop splitFactor)))
                     	(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum splitFactor)))
                     )
                 	(Loop (+ ?loopL "Split") (MNum splitFactor))
                 	?stride
                )
            )
         	(Loop (+ ?loopL "Split") (MNum splitFactor))
         	?stride
        )
     	(Loop ?loopL (MNum (/ ?loop splitFactor)))
    	(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum splitFactor)))
    )
 	:when ((> ?loop splitFactor) (= (% ?loop splitFactor) 0))
)
(rewrite
 	(LoopOut (Binary ?spbin (LoopIn ?a (Loop ?loopL (MNum ?loop)) ?strideA) (LoopIn ?b (Loop ?loopL (MNum ?loop)) ?strideB)) (Loop ?loopL (MNum ?loop)) ?stride)
 	(LoopOut
     	(LoopOut
         	(Binary ?spbin
            	(LoopIn
                 	(LoopIn
                     	?a
                     	(Loop ?loopL (MNum (/ ?loop splitFactor)))
                     	(MReplace ?strideA (MVar "z") (MMul (MVar "z") (MNum splitFactor)))
                    )
                 	(Loop (+ ?loopL "Split") (MNum splitFactor))
                 	?strideA
                )
                (LoopIn
                 	(LoopIn
                     	?b
                     	(Loop ?loopL (MNum (/ ?loop splitFactor)))
                     	(MReplace ?strideB (MVar "z") (MMul (MVar "z") (MNum splitFactor)))
                    )
                 	(Loop (+ ?loopL "Split") (MNum splitFactor))
                 	?strideB
                )
            )
         	(Loop (+ ?loopL "Split") (MNum splitFactor))
         	?stride
        )
     	(Loop ?loopL (MNum (/ ?loop splitFactor)))
    	(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum splitFactor)))
    )
 	:when ((> ?loop splitFactor) (= (% ?loop splitFactor) 0))
)
(rewrite ; SHORTCUT: immediately split loops with body of Binary(Binary(a, b), c)
 	(LoopOut
  		(Binary ?bin2
    		(Binary ?bin1
	    		(LoopIn ?a (Loop ?loopL (MNum ?loop)) ?strideA)
	      		(LoopIn ?b (Loop ?loopL (MNum ?loop)) ?strideB)
	        )
	        (LoopIn ?c (Loop ?loopL (MNum ?loop)) ?strideC)
        )
    (Loop ?loopL (MNum ?loop)) ?stride)
 	(LoopOut
     	(LoopOut
         	(Binary ?bin2
          		(Binary ?bin1
	            	(LoopIn
	                 	(LoopIn
	                     	?a
	                     	(Loop ?loopL (MNum (/ ?loop splitFactor)))
	                     	(MReplace ?strideA (MVar "z") (MMul (MVar "z") (MNum splitFactor)))
	                    )
	                 	(Loop (+ ?loopL "Split") (MNum splitFactor))
	                 	?strideA
	                )
	                (LoopIn
	                 	(LoopIn
	                     	?b
	                     	(Loop ?loopL (MNum (/ ?loop splitFactor)))
	                     	(MReplace ?strideB (MVar "z") (MMul (MVar "z") (MNum splitFactor)))
	                    )
	                 	(Loop (+ ?loopL "Split") (MNum splitFactor))
	                 	?strideB
	                )
	            )
				(LoopIn
                 	(LoopIn
                     	?c
                     	(Loop ?loopL (MNum (/ ?loop splitFactor)))
                     	(MReplace ?strideC (MVar "z") (MMul (MVar "z") (MNum splitFactor)))
                    )
                 	(Loop (+ ?loopL "Split") (MNum splitFactor))
                 	?strideC
                )
			)
         	(Loop (+ ?loopL "Split") (MNum splitFactor))
         	?stride
        )
     	(Loop ?loopL (MNum (/ ?loop splitFactor)))
    	(MReplace ?stride (MVar "z") (MMul (MVar "z") (MNum splitFactor)))
    )
 	:when ((> ?loop 512) (= (% ?loop splitFactor) 0))
)

; Swap dimensions
(rewrite  ; 0-1
	(LoopOut (LoopOut ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
	(LoopOut (LoopOut (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?outerLoop ?outerLoopAmt) ?outerSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt)
)
(rewrite  ; 0-2 SHORTCUT
	(LoopOut (LoopOut (LoopOut ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) ?midLoop ?midSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
	(LoopOut (LoopOut (LoopOut (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?outerLoop ?outerLoopAmt) ?outerSt) ?midLoop ?midSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt)
)
(rewrite ; 0-1
	(SwapLoops (LoopIn (LoopIn ?body (Loop ?outerLoop ?outerLoopAmt) ?outerSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt) ?innerLoop ?outerLoop)
	(LoopIn (LoopIn ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
)
(rewrite ; 0-2 SHORTCUT
	(SwapLoops (LoopIn (LoopIn (LoopIn ?body (Loop ?outerLoop ?outerLoopAmt) ?outerSt) ?midLoop ?midSt) (Loop ?innerLoop ?innerLoopAmt) ?innerSt) ?innerLoop ?outerLoop)
	(LoopIn (LoopIn (LoopIn ?body (Loop ?innerLoop ?innerLoopAmt) ?innerSt) ?midLoop ?midSt) (Loop ?outerLoop ?outerLoopAmt) ?outerSt)
)
(rewrite
	(SwapLoops (LoopIn ?body (Loop ?otherLoop ?otherLoopAmt) ?otherSt) ?innerLoop ?outerLoop)
	(LoopIn (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?otherLoop ?otherLoopAmt) ?otherSt)
	:when ((!= ?innerLoop ?otherLoop))
)
(rewrite
	(SwapLoops (LoopOut ?body (Loop ?otherLoop ?otherLoopAmt) ?otherSt) ?innerLoop ?outerLoop)
	(LoopOut (SwapLoops ?body ?innerLoop ?outerLoop) (Loop ?otherLoop ?otherLoopAmt) ?otherSt)
)
(rewrite
	(SwapLoops (Unary ?un ?body) ?innerLoop ?outerLoop)
	(Unary ?un (SwapLoops ?body ?innerLoop ?outerLoop))
)
(rewrite
	(SwapLoops (Binary ?bin ?bodyA ?bodyB) ?innerLoop ?outerLoop)
	(Binary ?bin (SwapLoops ?bodyA ?innerLoop ?outerLoop) (SwapLoops ?bodyB ?innerLoop ?outerLoop))
)

; Exp-Sum-Max Online Trick
(rewrite
	(LoopOut (Add (Exp (Add
		?online
		(Neg (LoopIn (LoopOut (Max (LoopIn ?maxAcc ?loop (MAccum)) ?online) ?loop (MAccum)) ?loop (MNum 0)))
	)) (LoopIn ?sumAcc ?loop (MAccum))) ?loop (MAccum))
	(LoopOut
		(Add
			(Mul
				(LoopIn ?sumAcc ?loop (MAccum))
				(Exp (Add (LoopIn ?maxAcc ?loop (MAccum)) (Neg (Max (LoopIn ?maxAcc ?loop (MAccum)) ?online))))
			)
			(Exp (Add ?online (Neg (Max (LoopIn ?maxAcc ?loop (MAccum)) ?online))))
		)
	?loop (MAccum))
)

; Online softmax -> weighted sum trick
(rewrite
	(LoopOut
		(LoopOut
			(Add
				(Mul
					(LoopIn
						(Mul
							(Exp
								(Add
									(LoopIn (LoopOut ?dot ?kLoop (MVar "z")) ?kLoop (MVar "z"))
									(Neg (LoopIn (LoopOut ?max_acc ?kLoop (MAccum)) ?kLoop (MNum 0)))
								)
							)
							(Recip (LoopIn (LoopOut (Add (Mul ?exp_sum_acc (Exp (Add ?redot (Neg ?renew_max)))) ?weight) ?kLoop (MAccum)) ?kLoop (MNum 0)))
						)
						?vLoop (MNum 0)
					)
					(LoopIn (LoopIn ?val ?kLoop ?valRow) ?vLoop (MVar "z"))
				)
				(LoopIn (LoopIn ?output_acc ?kLoop (MAccum)) ?vLoop (MVar "z"))
			)
			?vLoop (MVar "z")
		)
		?kLoop (MAccum)
	)
	(LoopOut
		(Mul
			(LoopIn
				(LoopOut
					(LoopOut
						(Add
							(Mul
								(LoopIn (LoopIn ?output_acc ?kLoop (MAccum)) ?vLoop (MVar "z"))
								(LoopIn (Exp (Add ?redot (Neg ?renew_max))) ?vLoop (MNum 0))
							)
							(Mul
								(LoopIn ?weight ?vLoop (MNum 0))
								(LoopIn (LoopIn ?val ?kLoop ?valRow) ?vLoop (MVar "z"))
							)
						)
						?vLoop (MVar "z")
					)
					?kLoop (MAccum)
				)
				?vLoop (MVar "z")
			)
			(Recip (LoopIn (LoopOut (Add (Mul ?exp_sum_acc (Exp (Add ?redot (Neg ?renew_max)))) ?weight) ?kLoop (MAccum)) ?vLoop (MNum 0)))
		)
		?vLoop (MVar "z")
	)
)



; ───────────────── TESTS ─────────────────
; Common variables
(let tensorA (Tensor "A"))
(let tensorB (Tensor "B"))
(let tensorC (Tensor "C"))
(let strideOne (MVar "z"))

; ───────────────── Fission test (1 loop -> 3 sequential loops) ─────────────────
(push)
(let loop (Loop "l" (MNum 4096)))
(let full (LoopOut (Exp (Sin (Add (LoopIn tensorA loop strideOne) (Sin (LoopIn tensorB loop strideOne))))) loop strideOne))
(run 10)
(let part0 (LoopOut (Sin (LoopIn tensorB loop strideOne)) loop strideOne))
(let part1 (LoopOut (Add (LoopIn tensorA loop strideOne) (LoopIn part0 loop strideOne)) loop strideOne))
(let part2 (LoopOut (Sin (LoopIn part1 loop strideOne)) loop strideOne))
(let part3 (LoopOut (Exp (LoopIn part2 loop strideOne)) loop strideOne))
(check (= full part3))
(pop)

; ───────────────── Fusion test (3 sequential loops -> 1 loop) ─────────────────
(push)
(let loop (Loop "l" (MNum 4096)))
(let part1 (LoopOut (Sin (LoopIn tensorA loop strideOne)) loop strideOne))
(let part2 (LoopOut (Add (LoopIn part1 loop strideOne) (LoopIn tensorB loop strideOne)) loop strideOne))
(let part3 (LoopOut (Exp (LoopIn part2 loop strideOne)) loop strideOne))
(run 10)
(let full (LoopOut (Exp (Add (Sin (LoopIn tensorA loop strideOne)) (LoopIn tensorB loop strideOne))) loop strideOne))
(check (= part3 full))
(pop)

; ───────────────── Loop merging test (2 nested loops -> 1 merged loop) ─────────────────
(push)
(let mLoop (Loop "m" (MNum 4096)))
(let nLoop (Loop "n" (MNum 4096)))
(let mStride (MMul (MVar "z") (MNum 4096)))
(let nStride (MVar "z"))
(let aStrided (LoopIn (LoopIn tensorA mLoop mStride) nLoop nStride))
(let bStrided (LoopIn (LoopIn tensorB mLoop mStride) nLoop nStride))
(let cStrided (LoopIn (LoopIn tensorC mLoop mStride) nLoop nStride))
(let body (Sin (Add (Exp (Add aStrided (Sin bStrided))) cStrided)))
(let out (LoopOut (LoopOut body nLoop nStride) mLoop mStride))
(run 12)
; rename loops to dummy names
(rewrite (Loop ?_id ?trip) (Loop "_a" ?trip))
(run 1)

(let newStride (MVar "z"))
(let newLoop (Loop "_a" (MNum 16777216)))
(let newaStrided (LoopIn tensorA newLoop newStride))
(let newbStrided (LoopIn tensorB newLoop newStride))
(let newcStrided (LoopIn tensorC newLoop newStride))
(let newbody (Sin (Add (Exp (Add newaStrided (Sin newbStrided))) newcStrided)))
(let newout (LoopOut newbody newLoop newStride))
(check (= out newout))
(pop)

; ───────────────── Loop swapping test ─────────────────
(push)
(let mLoop (Loop "m" (MNum 4096)))
(let nLoop (Loop "n" (MNum 10)))
(let mStride (MMul (MVar "z") (MNum 10)))
(let nStride (MVar "z"))
(let aStrided (LoopIn (LoopIn tensorA mLoop mStride) nLoop nStride))
(let bStrided (LoopIn (LoopIn tensorB mLoop mStride) nLoop nStride))
(let body (Sin aStrided))
(let out (LoopOut (LoopOut body nLoop nStride) mLoop mStride))
(run 5)
; rename loops to dummy names
(rewrite (Loop ?_id ?trip) (Loop "_a" ?trip))
(run 1)
(let swapaStrided (LoopIn (LoopIn tensorA nLoop nStride) mLoop mStride))
(let swapbStrided (LoopIn (LoopIn tensorB nLoop nStride) mLoop mStride))
(let swapbody (Sin swapaStrided))
(let swapout (LoopOut (LoopOut swapbody mLoop mStride) nLoop nStride))
(check (= out swapout))
(pop)

; ───────────────── Loop splitting test (1 merged loop -> 2 nested loops) ─────────────────
(push)
(let newStride (MVar "z"))
(let newLoop (Loop "m" (MNum 4096)))
(let newaStrided (LoopIn tensorA newLoop newStride))
(let newbStrided (LoopIn tensorB newLoop newStride))
(let newcStrided (LoopIn tensorC newLoop newStride))
(let newbody (Sin newbStrided))
(let newout (LoopOut newbody newLoop newStride))

(run 13)
; rename loops to dummy names
(rewrite (Loop ?_id ?trip) (Loop "_a" ?trip))
(run 1)

(let loop1 (Loop "_a" (MNum 64)))
(let loop2 (Loop "_a" (MNum 64)))
(let stride1 (MMul (MVar "z") (MNum 64)))
(let stride2 (MVar "z"))
(let aStrided (LoopIn (LoopIn tensorA loop1 stride1) loop2 stride2))
(let bStrided (LoopIn (LoopIn tensorB loop1 stride1) loop2 stride2))
(let cStrided (LoopIn (LoopIn tensorC loop1 stride1) loop2 stride2))
(let body (Sin bStrided))
(let out (LoopOut (LoopOut body loop2 stride2) loop1 stride1))
(check (= out newout))
(pop)

; ───────────────── online softmax ─────────────────
(push)
(let qLoop (Loop "q" (MNum 4096)))   ; query row
(let kLoop (Loop "k" (MNum 4096)))   ; key   row
(let scores (Tensor "scores"))
(let row64  (MMul (MVar "z") (MNum 64)))  ; +64  → next row
(let col1   (MVar "z"))                   ; +1   → next column

; standard softmax
; compute max (Pass 1)
(let scoresIn (LoopIn (LoopIn scores qLoop row64) kLoop col1))
(let maxAccum (LoopIn (LoopIn (NewAcc 0) qLoop col1) kLoop (MAccum)))
(let max (LoopOut (LoopOut (Max maxAccum scoresIn) kLoop (MAccum)) qLoop col1))

; compute sum of exponents (Pass 2)
(let maxIn (LoopIn (LoopIn max qLoop col1) kLoop (MNum 0)))
(let expSumAccum (LoopIn (LoopIn (NewAcc 0) qLoop col1) kLoop (MAccum)))
(let expSum (LoopOut (LoopOut (Add (Exp (Add scoresIn (Neg maxIn))) expSumAccum) kLoop (MAccum)) qLoop col1))

; compute final values (Pass 3)
(let expSumIn (LoopIn (LoopIn expSum qLoop col1) kLoop (MNum 0)))
(let body (Mul (Exp (Add scoresIn (Neg maxIn))) (Recip expSumIn)))
(let out (LoopOut (LoopOut body kLoop col1) qLoop row64))
(run 4)

; online softmax
; compute max and sum of exponents (Pass 1)
(let oscoresIn (LoopIn (LoopIn scores qLoop row64) kLoop col1))
(let omaxAccum (LoopIn (LoopIn (NewAcc 0) qLoop col1) kLoop (MAccum)))
(let oexpSumAccum (LoopIn (LoopIn (NewAcc 0) qLoop col1) kLoop (MAccum)))
(let onewMax (Max omaxAccum oscoresIn))
(let onewSum (Add (Mul oexpSumAccum (Exp (Add omaxAccum (Neg onewMax)))) (Exp (Add oscoresIn (Neg onewMax)))))
(let omax (LoopOut (LoopOut onewMax kLoop (MAccum)) qLoop col1))
(let oexpSum (LoopOut (LoopOut onewSum kLoop (MAccum)) qLoop col1))

; compute final values (Pass 2)
(let oexpSumIn (LoopIn (LoopIn oexpSum qLoop col1) kLoop (MNum 0)))
(let omaxIn (LoopIn (LoopIn omax qLoop col1) kLoop (MNum 0)))
(let obody (Mul (Exp (Add oscoresIn (Neg omaxIn))) (Recip oexpSumIn)))
(let oout (LoopOut (LoopOut obody kLoop col1) qLoop row64))
(check (= oout out))
(pop)

; ───────────────── flash attention ─────────────────
(push)

; loops
(let qLoop (Loop "q" (MNum 4096)))   ; query row
(let kLoop (Loop "k" (MNum 4096)))   ; key   row
(let dLoop (Loop "d" (MNum 64)))     ; dot‑prod dim
(let vLoop (Loop "v" (MNum 64)))     ; value dim

; tensors & strides
(let tensorQ (Tensor "Q"))           ; [4096 × 64] row‑major
(let tensorK (Tensor "K"))
(let tensorV (Tensor "V"))
(let row64  (MMul (MVar "z") (MNum 64)))  ; +64  → next row
(let row4096 (MMul (MVar "z") (MNum 4096)))
(let col1   (MVar "z"))                   ; +1   → next column
(let skip (MNum 0))

; accumulators
(let dot_acc (LoopIn (LoopIn (LoopIn (NewAcc 0) qLoop row4096) kLoop col1) dLoop (MAccum)))
(let score_max_acc (LoopIn (LoopIn (NewAcc 0) qLoop col1) kLoop (MAccum)))
(let exp_sum_acc (LoopIn (LoopIn (NewAcc 0) qLoop col1) kLoop (MAccum)))
(let output_acc (LoopIn (LoopIn (LoopIn (NewAcc 0) qLoop row64) kLoop (MAccum)) vLoop col1))

; inputs
(let q_in (LoopIn (LoopIn (LoopIn tensorQ qLoop row64) kLoop skip) dLoop col1))
(let k_in (LoopIn (LoopIn (LoopIn tensorK qLoop skip) kLoop col1) dLoop row64))
(let v_in (LoopIn (LoopIn (LoopIn tensorV qLoop skip) kLoop row64) vLoop col1))

; ──── naive ────

; get dot products
(let dots (LoopOut (LoopOut (LoopOut (Add (Mul q_in k_in) dot_acc) dLoop (MAccum)) kLoop col1) qLoop row4096))

; get max
(let dots_in (LoopIn (LoopIn dots qLoop row4096) kLoop col1))
(let max (LoopOut (LoopOut (Max score_max_acc dots_in) kLoop (MAccum)) qLoop col1))

; get exp sum
(let max_in (LoopIn (LoopIn max qLoop col1) kLoop skip))
(let exp_sum (LoopOut (LoopOut (Add (Exp (Add dots_in (Neg max_in))) exp_sum_acc) kLoop (MAccum)) qLoop col1))

; get final scores
(let exp_sum_in (LoopIn (LoopIn exp_sum qLoop col1) kLoop skip))
(let final_scores (LoopOut (LoopOut (Mul (Exp (Add dots_in (Neg max_in))) (Recip exp_sum_in)) kLoop col1) qLoop row4096))

; get output
(let final_scores_in (LoopIn (LoopIn (LoopIn final_scores qLoop row4096) kLoop col1) vLoop skip))
(let output (LoopOut (LoopOut (LoopOut (Add (Mul v_in final_scores_in) output_acc) vLoop col1) kLoop (MAccum)) qLoop row64))

(run 12)

; ──── online softmax ────

; get dot products
(let online_dots (LoopOut (LoopOut (LoopOut (Add (Mul q_in k_in) dot_acc) dLoop (MAccum)) kLoop col1) qLoop row4096))

; get max and exp_sum
(let online_dots_in (LoopIn (LoopIn online_dots qLoop row4096) kLoop col1))
(let online_new_max (Max score_max_acc dots_in))
(let online_rescale (Exp (Add score_max_acc (Neg online_new_max))))
(let online_weight (Exp (Add online_dots_in (Neg online_new_max))))
(let online_exp_sum (LoopOut (LoopOut (Add (Mul exp_sum_acc online_rescale) online_weight) kLoop (MAccum)) qLoop col1))
(let online_max (LoopOut (LoopOut online_new_max kLoop (MAccum)) qLoop col1))

; get final scores
(let online_exp_sum_in (LoopIn (LoopIn online_exp_sum qLoop col1) kLoop skip))
(let online_max_in (LoopIn (LoopIn online_max qLoop col1) kLoop skip))
(let online_final_scores (LoopOut (LoopOut (Mul (Exp (Add online_dots_in (Neg online_max_in))) (Recip online_exp_sum_in)) kLoop col1) qLoop row4096))

; get val weighted sum
(let online_final_scores_in (LoopIn (LoopIn (LoopIn online_final_scores qLoop row4096) kLoop col1) vLoop skip))
(let online_output (LoopOut (LoopOut (LoopOut (Add (Mul v_in online_final_scores_in) output_acc) vLoop col1) kLoop (MAccum)) qLoop row64))

(check (= output online_output))

; ──── fused kv-online softmax ────

; get dot products, max, and exp_sum
(let kv_dot (LoopOut (Add (Mul q_in k_in) dot_acc) dLoop (MAccum)))
(let kv_new_max (Max score_max_acc kv_dot))
(let kv_rescale (Exp (Add score_max_acc (Neg kv_new_max))))
(let kv_weight (Exp (Add kv_dot (Neg kv_new_max))))
(let kv_exp_sum (LoopOut (LoopOut (Add (Mul exp_sum_acc kv_rescale) kv_weight) kLoop (MAccum)) qLoop col1))
(let kv_max (LoopOut (LoopOut kv_new_max kLoop (MAccum)) qLoop col1))
(let kv_dots (LoopOut (LoopOut kv_dot kLoop col1) qLoop row4096))

; get final scores
(let kv_exp_sum_in (LoopIn (LoopIn kv_exp_sum qLoop col1) kLoop skip))
(let kv_max_in (LoopIn (LoopIn kv_max qLoop col1) kLoop skip))
(let kv_dots_in (LoopIn (LoopIn kv_dots qLoop row4096) kLoop col1))
(let kv_final_scores (LoopOut (LoopOut (Mul (Exp (Add kv_dots_in (Neg kv_max_in))) (Recip kv_exp_sum_in)) kLoop col1) qLoop row4096))

; get val weighted sum
(let kv_final_scores_in (LoopIn (LoopIn (LoopIn kv_final_scores qLoop row4096) kLoop col1) vLoop skip))
(let kv_output (LoopOut (LoopOut (LoopOut (Add (Mul v_in kv_final_scores_in) output_acc) vLoop col1) kLoop (MAccum)) qLoop row64))

(check (= output kv_output))
(check (= online_output kv_output))

; ──── fused score-val-online softmax ────

; get dot products, max, and exp_sum
(let val_dot (LoopOut (Add (Mul q_in k_in) dot_acc) dLoop (MAccum)))
(let val_new_max (Max score_max_acc val_dot))
(let val_rescale (Exp (Add score_max_acc (Neg val_new_max))))
(let val_weight (Exp (Add val_dot (Neg val_new_max))))
(let val_exp_sum (LoopOut (LoopOut (Add (Mul exp_sum_acc val_rescale) val_weight) kLoop (MAccum)) qLoop col1))
(let val_max (LoopOut (LoopOut val_new_max kLoop (MAccum)) qLoop col1))
(let val_dots (LoopOut (LoopOut val_dot kLoop col1) qLoop row4096))

; get final scores and val weighted sum
(let val_exp_sum_in (LoopIn (LoopIn val_exp_sum qLoop col1) kLoop skip))
(let val_max_in (LoopIn (LoopIn val_max qLoop col1) kLoop skip))
(let val_dots_in (LoopIn (LoopIn val_dots qLoop row4096) kLoop col1))
(let val_final_score (Mul (Exp (Add val_dots_in (Neg val_max_in))) (Recip val_exp_sum_in)))
(let val_final_score_b (LoopIn val_final_score vLoop skip))
(let val_output (LoopOut (LoopOut (LoopOut (Add (Mul val_final_score_b v_in) output_acc) vLoop col1) kLoop (MAccum)) qLoop row64))

(check (= output val_output))
(check (= online_output val_output))
(check (= kv_output val_output))

; ──── one-pass ────

(let one_dot (LoopOut (Add (Mul q_in k_in) dot_acc) dLoop (MAccum)))
(let one_new_max (Max score_max_acc one_dot))
(let one_rescale (Exp (Add score_max_acc (Neg one_new_max))))
(let one_weight (Exp (Add one_dot (Neg one_new_max))))
(let one_exp_sum_new (Add (Mul exp_sum_acc one_rescale) one_weight))
(let one_weight_b (LoopIn one_weight vLoop skip))
(let one_rescale_b (LoopIn one_rescale vLoop skip))
(let one_partial_output (LoopOut (LoopOut (Add (Mul output_acc one_rescale_b) (Mul one_weight_b v_in)) vLoop col1) kLoop (MAccum)))
(let one_exp_sum (LoopOut one_exp_sum_new kLoop (MAccum)))
(let one_exp_sum_b (LoopIn one_exp_sum vLoop skip))
(let one_output (LoopOut (LoopOut (Mul (LoopIn one_partial_output vLoop col1) (Recip one_exp_sum_b)) vLoop col1) qLoop row64))

(check (= output one_output)) ; naive -> one-loop
(check (= online_output one_output)) ; online-softmax -> one-loop
(check (= kv_output one_output)) ; kv-fused online softmax -> one-loop
(check (= val_output one_output)) ; val-sum-fused online softmax -> one-loop