-- Definition of the left justified heap tree
data Heap n = Leaf n | Branch (Maybe n, (Maybe (Heap n), Maybe (Heap n)))
              deriving Show

-- Return an empty min heap
empty' :: Heap n
empty' = Branch (Nothing, (Nothing, Nothing))

-- Return the length of the heap
length' :: Heap n -> Int
length' (Leaf _) = 1
length' (Branch (Nothing, (Nothing, Nothing))) = 0
length' (Branch (Just _, (Just l, Nothing))) = -1   -- Subtract 1 because heap is not full
length' (Branch (Just _, (Just l, Just r))) = 1 + max (length' l) (length' r)

-- v: vertex, nv: new vertex, l: left, r: right
insert' :: Ord n => Heap n -> n -> Heap n
insert' (Branch (Nothing, (Nothing, Nothing))) nv = Leaf nv
-- Heap has only one left leaf
insert' (Branch (Just v, (Just (Leaf l), Nothing))) nv
    | nv < v = Branch (Just nv, (Just (Leaf l), Just (Leaf v)))
    | otherwise = Branch (Just v, (Just (Leaf l), Just (Leaf nv)))
insert' (Branch (Just v, (Just l, Just r))) nv
    | nv < v = insert' (Branch (Just nv, (Just l, Just r))) v
    | length' l > length' r = Branch (Just v, (Just l, Just $ insert' r nv))
    | otherwise = Branch (Just v, (Just $ insert' l nv, Just r))
-- Insert under a leaf
insert' (Leaf v) nv
    | nv < v = Branch (Just nv, (Just $ Leaf v, Nothing))
    | otherwise = Branch (Just v, (Just $ Leaf nv, Nothing))

fromList' :: Ord n => [n] -> Heap n
fromList' = foldl insert' empty'

lookup' :: Ord n => n -> Heap n -> Int
lookup' _ (Branch (Nothing, (Nothing, Nothing))) = 0
lookup' n (Leaf v) = if n==v then 1 else 0
lookup' n (Branch (Just v, (Just l, Nothing))) = if n==v then 1 else lookup' n l
lookup' n (Branch (Just v, (Just l, Just r))) = if n==v then 1 else lookup' n l + lookup' n r

maxElement' :: Ord n => Heap n -> Maybe n
maxElement' (Branch (Nothing, (Nothing, Nothing))) = Nothing
maxElement' (Leaf v) = Just v
maxElement' (Branch (Just v, (Just l, Nothing))) = max (Just v) (maxElement' l)
maxElement' (Branch (Just v, (Just l, Just r))) = max (Just v) (max (maxElement' l) (maxElement' r))

delete' :: Ord n => n -> Heap n -> Heap n
delete' n heap = if lookup' n heap == 1 then deleteElement' n heap else heap
    where
        deleteElement' n heap = heap    -- Fix here and implement element deletion
        

-- Extract the value of the branch vertex
extract' :: Ord n => Heap n -> Maybe n
extract' (Branch (v, (Just _, Just _))) = v
extract' (Branch (v, (Just _, Nothing))) = v

isValidMinHeap' :: Ord n => Heap n -> Int
isValidMinHeap' (Branch (Nothing, (Nothing, Nothing))) = 1
isValidMinHeap' (Leaf _) = 1
isValidMinHeap' (Branch (_, (Nothing, Nothing))) = 0    -- This is a leaf. A branch cant have two nothings
isValidMinHeap' (Branch (_, (Nothing, Just _))) = 0 -- Left justified heap cant have right elements before left
isValidMinHeap' (Branch (Just v, (Just (Leaf l), Nothing))) = if v < l then 1 else 0
isValidMinHeap' (Branch (Just v, (Just (Leaf l), Just (Leaf r)))) = if v < l && v < r then 1 else 0
isValidMinHeap' (Branch (v, (Just l, Just r))) = if v < lVal && v < rVal then min (isValidMinHeap' l) (isValidMinHeap' r) else 0
    where
        lVal = extract' l
        rVal = extract' r

main :: IO ()
main = do
    print $ Branch (Just 1,(Just (Branch (Just 3,(Just (Leaf 5),Just (Leaf 4)))),Just (Branch (Just 2,(Just (Leaf 6),Nothing)))))   -- Homework example heap
    let myHeap = fromList' [5, 1, 2, 4, 3, 6]
    print myHeap    -- Expect to be the same with the previous print statement

    print $ lookup' 4 myHeap    -- Expect 1
    print $ lookup' 10 myHeap   -- Expect 0
    print $ lookup' 10 empty'   -- Expect 0

    print $ maxElement' myHeap  -- Expect Just 6
    -- print $ maxElement' empty'  -- Expect Nothing but error during compilation

    print $ isValidMinHeap' myHeap  -- Expect 1
    -- print $ isValidMinHeap' empty' -- Expect 1 but error during compilation
    print $ isValidMinHeap' $ Branch (Just 2, (Just (Leaf 3), Just (Leaf 1)))   -- Expect 0
    print $ isValidMinHeap' $ Branch (Just 1,(Just (Branch (Just 0,(Just (Leaf 5),Just (Leaf 4)))),Just (Branch (Just 2,(Just (Leaf 6),Nothing))))) -- Expect 0
    print $ isValidMinHeap' $ Branch (Just 2, (Nothing, Nothing))   -- Expect 0
    print $ isValidMinHeap' $ Branch (Just 2, (Nothing, Just (Leaf 3)))   -- Expect 0
    print $ isValidMinHeap' $ Branch (Just 2, (Just (Leaf 1), Nothing))   -- Expect 0

    print $ delete' 4 myHeap   -- Expect Branch (Just 1,(Just (Branch (Just 4,(Just (Leaf 5),Nothing))),Just (Branch (Just 2,(Just (Leaf 6),Nothing)))))
    print $ delete' 10 myHeap   -- Expect myHeap untouched