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

-- lookup' :: Ord n => n -> Heap n -> Int

-- maxElement' Ord n => Heap n -> Maybe n

-- delete' :: Ord n => n -> Heap n -> Heap n

-- isValidMinHeap' :: Ord n => Heap n -> Int


main :: IO ()
main = do
    let myHeap = Branch
                (Just 1, (
                Just $ Branch (Just 3, (Just $ Leaf 5, Just $ Leaf 4)),
                Just $ Branch (Just 2, (Just $ Leaf 6, Nothing))
                ))
    print myHeap
    print $ fromList' [5, 1, 2, 4, 3, 6]