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

-- delete' :: Ord n => n -> Heap n -> Heap n

-- isValidMinHeap' :: Ord n => Heap n -> Int


main :: IO ()
main = do
    let myHeap = fromList' [5, 1, 2, 4, 3, 6]
    print myHeap
    print $ lookup' 10 myHeap
    -- print $ maxElement' (Branch (Nothing, (Nothing, Nothing)))