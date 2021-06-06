data Heap n = Leaf n | Branch (Maybe n, (Maybe (Heap n), Maybe (Heap n)))
              deriving Show

-- Return an empty min heap
empty' :: Heap n
empty' = Branch (Nothing, (Nothing, Nothing))

-- Return the depth of the heap
depth' :: Heap n -> Int
depth' (Leaf _) = 1
depth' (Branch (Nothing, (Nothing, Nothing))) = 0
depth' (Branch (Just _, (Nothing, Nothing))) = 1
depth' (Branch (Just _, (Just l, Nothing))) = 2
depth' (Branch (Just _, (Just l, Just r))) = 1 + max (depth' l) (depth' r)

-- v: vertex, nv: new vertex, l: left, r: right
insert' :: Ord n => Heap n -> n -> Heap n
insert' (Branch (Nothing, (Nothing, Nothing))) nv = Branch (Just nv, (Nothing, Nothing))
-- Insert under a leaf
insert' (Leaf v) nv
    | nv < v = Branch (Just nv, (Just $ Leaf v, Nothing))
    | otherwise = Branch (Just v, (Just $ Leaf nv, Nothing))
-- Heap has only one left leaf
insert' (Branch (Just v, (Just (Leaf l), Nothing))) nv
    | nv < v = Branch (Just nv, (Just (Leaf l), Just (Leaf v)))
    | otherwise = Branch (Just v, (Just (Leaf l), Just (Leaf nv)))
insert' (Branch (Just v, (Just l, Just r))) nv
    | nv < v = insert' (Branch (Just nv, (Just l, Just r))) v
    | otherwise = Branch (Just v, (Just l, Just $ insert' r nv))

-- fromList' :: Ord n => [n] -> Heap n

-- lookup' :: Ord n => n -> Heap n -> Int

-- maxElement' Ord n => Heap n -> Maybe n

-- delete' :: Ord n => n -> Heap n -> Heap n

-- isValidMinHeap' :: Ord n => Heap n -> Int


main :: IO ()
main = do
    let myHeap = Branch
                (Just 1, (
                Just $ Branch (Just 3, (Just $ Leaf 5, Just $ Leaf 4)),
                Just $ Leaf 2
                ))
    print $ insert' myHeap 0