-- Definition of the leftist heap tree
data Heap n = Leaf n | Branch (Maybe n, (Maybe (Heap n), Maybe (Heap n)))
              deriving Show

-- Return an empty min heap
empty' :: Heap n
empty' = Branch (Nothing, (Nothing, Nothing))   -- Given definition of an empty min heap

-- Insert new elements to the heap
-- v: vertex, nv: new vertex, l: left, r: right
insert' :: Ord n => Heap n -> n -> Heap n
-- If the heap is empty, initialize it with a leaf
insert' (Branch (Nothing, (Nothing, Nothing))) nv = Leaf nv
-- Insert under a leaf
insert' (Leaf v) nv
    | nv < v = Branch (Just nv, (Just $ Leaf v, Nothing))
    | otherwise = Branch (Just v, (Just $ Leaf nv, Nothing))
-- Node has only one left leaf
insert' (Branch (Just v, (Just (Leaf l), Nothing))) nv
    | nv < v = Branch (Just nv, (Just (Leaf l), Just (Leaf v)))
    | otherwise = Branch (Just v, (Just (Leaf l), Just (Leaf nv)))
-- Node has both branches
insert' (Branch (Just v, (Just l, Just r))) nv
    | nv < v = insert' (Branch (Just nv, (Just l, Just r))) v
    | length' l > length' r = Branch (Just v, (Just l, Just $ insert' r nv))
    | otherwise = Branch (Just v, (Just $ insert' l nv, Just r))

{-
    Similar checks (Empty heap, Leaf, Node with left branch, Node with both branches) are defined
        for lenght, toList, removeFromList, lookup, maxElement, extract and isValidMinHeap funtions
        I will not be explicitly commenting for each function
-}

-- Return the length of the heap
-- Similar to the rank of the tree and is a required check during insertion
length' :: Ord n => Heap n -> Int
length' (Leaf _) = 1
length' (Branch (Nothing, (Nothing, Nothing))) = 0
length' (Branch (Just _, (Just l, Nothing))) = -1   -- Subtract 1 because heap is not full
length' (Branch (Just _, (Just l, Just r))) = 1 + max (length' l) (length' r)

-- Construct a min heap from a list
fromList' :: Ord n => [n] -> Heap n
fromList' = foldl insert' empty'

-- Convert a min heap to a list
toList' :: Ord n => Heap n -> [n]
toList' (Leaf v) = [v]
toList' (Branch (Nothing, (_, _))) = []
toList' (Branch (Just v, (Just l, Nothing))) = v:toList' l
toList' (Branch (Just v, (Just l, Just r))) = v:(toList' l ++ toList' r)

-- Remove an element from list
-- This function is defined to remove a specific node from the heap
removeFromList' :: Ord n => n -> [n] -> [n]
removeFromList' _ [] = []
removeFromList' x (y:ys)
    | x == y = removeFromList' x ys
    | otherwise = y:removeFromList' x ys

-- Check if a node exists in the heap
-- If it exists return 1, if not return 0
lookup' :: Ord n => n -> Heap n -> Int
lookup' _ (Branch (Nothing, (Nothing, Nothing))) = 0
lookup' n (Leaf v) = if n==v then 1 else 0
lookup' n (Branch (Just v, (Just l, Nothing))) = if n==v then 1 else lookup' n l
lookup' n (Branch (Just v, (Just l, Just r))) = if n==v then 1 else lookup' n l + lookup' n r

-- Get the maximum element of the min heap
maxElement' :: Ord n => Heap n -> Maybe n
maxElement' (Branch (Nothing, (Nothing, Nothing))) = Nothing
maxElement' (Leaf v) = Just v
maxElement' (Branch (Just v, (Just l, Nothing))) = max (Just v) (maxElement' l)
maxElement' (Branch (Just v, (Just l, Just r))) = max (Just v) (max (maxElement' l) (maxElement' r))

-- Deconstruct the heap into a list, remove element from list and return a reconstructed new min heap
deleteList' :: Ord n => n -> Heap n -> Heap n
deleteList' n heap = fromList' $ removeFromList' n (toList' heap)

-- Delete the element being searched from the heap
-- If the element is not in the heap return the heap without modification
delete' :: Ord n => n -> Heap n -> Heap n
delete' n heap = if lookup' n heap == 1 then deleteList' n heap else heap

-- Extract the value (Maybe n) of the branch vertex for comparisons
extract' :: Ord n => Heap n -> Maybe n
extract' (Leaf v) = Just v
extract' (Branch (v, (Just _, Just _))) = v
extract' (Branch (v, (Just _, Nothing))) = v

-- Check if the heap is a valid minimum heap or not
-- If the heap is a valid minimum heap return 1 else return 0
isValidMinHeap' :: Ord n => Heap n -> Int
isValidMinHeap' (Branch (Nothing, (Nothing, Nothing))) = 1  -- Empty heap is a valid minimum heap
isValidMinHeap' (Leaf _) = 1
isValidMinHeap' (Branch (_, (Nothing, Nothing))) = 0    -- This is a leaf. A branch cant have two nothings
isValidMinHeap' (Branch (_, (Nothing, Just _))) = 0 -- Left justified heap cant have right elements before left
isValidMinHeap' (Branch (Just v, (Just (Leaf l), Nothing))) = if v < l then 1 else 0
isValidMinHeap' (Branch (Just v, (Just (Leaf l), Just (Leaf r)))) = if v < l && v < r then 1 else 0
isValidMinHeap' (Branch (v, (Just l, Just r))) = if v < lVal && v < rVal then min (isValidMinHeap' l) (isValidMinHeap' r) else 0
    where
        -- Extract values of nodes to compare them
        -- Eq instance for the Heap can be derived in order to compare values and avoid value extraction here
        lVal = extract' l
        rVal = extract' r

{-
    Below, functions used during deletion of a node at a specific location can be observed
    They first delete the node, insert the bottom right node to its place then heapify
    It should be noted that even though they are not being used, they are there to
        show the process of deleting a node at a specific location. They partially function
        yet they are not being utilized because a working `delete` function is defined
-}

-- Returns the bottom right element of the tree for substitution
botRight' :: Ord n => Heap n -> Heap n
botRight' k@(Leaf v) = k
botRight' (Branch (Just v, (Just l, Nothing))) = botRight' l
botRight' (Branch (Just v, (Just l, Just r)))
    | depth' l > depth' r = botRight' l
    | otherwise = botRight' r

-- Return the depth of the heap
-- Required for joining two trees
depth' :: Ord n => Heap n -> Int
depth' (Leaf _) = 1
depth' (Branch (Nothing, (Nothing, Nothing))) = 0
depth' (Branch (Just _, (Just l, Nothing))) = 1 + depth' l
depth' (Branch (Just _, (Just l, Just r))) = 1 + max (depth' l) (depth' r)

-- Extract the pure value (n) of a Maybe n
-- Needed for recreation of the heap and is fragile (does not check Nothing)
extractPure' :: Ord n => Maybe n -> n
extractPure' (Just v) = v

-- Merges two heaps of varying lengths
-- k1: heap 1, k2: heap 2
merge' :: Ord n => Heap n -> Heap n -> Heap n
merge' k1 (Leaf v) = insert' k1 v
merge' (Leaf v) k2 = insert' k2 v
merge' k1@(Branch (Just v1, (Just l1, Nothing))) k2@(Branch (Just v2, (Just l2, Nothing)))
    | v2 > v1 = join' v2 l2 (merge' k1 empty')
    | otherwise = join' v1 l1 (merge' empty' k2)
merge' k1@(Branch (Just v1, (Just l1, Just r1))) k2@(Branch (Just v2, (Just l2, Nothing)))
    | v2 > v1 = join' v2 l2 (merge' k1 empty')
    | otherwise = join' v1 l1 (merge' r1 k2)
merge' k1@(Branch (Just v1, (Just l1, Nothing))) k2@(Branch (Just v2, (Just l2, Just r2)))
    | v2 > v1 = join' v2 l2 (merge' k1 r2)
    | otherwise = join' v1 l1 (merge' empty' k2)
merge' k1@(Branch (Just v1, (Just l1, Just r1))) k2@(Branch (Just v2, (Just l2, Just r2)))
    | v2 > v1 = join' v2 l2 (merge' k1 r2)
    | otherwise = join' v1 l1 (merge' r1 k2)

-- Joins two heaps with a node
join' :: Ord n => n -> Heap n -> Heap n -> Heap n
join' v k1 k2
    | depth' k2 < depth' k1 = Branch (Just v, (Just k2, Just k1))
    | otherwise = Branch (Just v, (Just k1, Just k2))

-- Removes the min element from the heap
-- Same as extractMin in a min-heap data structure
deleteMin' :: Ord n => Heap n -> Heap n
deleteMin' (Leaf _) = empty'
deleteMin' (Branch (Just v, (Just l, Nothing))) = deleteMin' l
deleteMin' (Branch (Just v, (Just l, Just r))) = merge' l r

-- Find the node (branch or leaf) to be deleted in the heap and delete it
getNode' :: Ord n => n -> Heap n -> Heap n
getNode' _ k@(Leaf _) = deleteMin' k
getNode' x k@(Branch (Just v, (Just l, Nothing)))
    | x == v = deleteMin' k
    | otherwise = getNode' x l
getNode' x k@(Branch (Just v, (Just l, Just r)))
    | x == v = deleteMin' k
    | lookup' x l == 1 = getNode' x l
    | otherwise = getNode' x r

-- The second delete function that deletes the node, inserts the bottom right and heapifies
delete2' :: Ord n => n -> Heap n -> Heap n
delete2' n heap = if lookup' n heap == 1 then getNode' n heap else heap

{-
    The second delete function ends here
    Below is the main function that checks base and edge cases of the defined functions
-}

main :: IO ()
main = do
    -- Print the heap defined in the homework
    print $ Branch (Just 1,(Just (Branch (Just 3,(Just (Leaf 5),Just (Leaf 4)))),Just (Branch (Just 2,(Just (Leaf 6),Nothing)))))   -- Homework example heap
    let myHeap = fromList' [5, 1, 2, 4, 3, 6]
    print myHeap    -- Expect to be the same with the previous print statement

    -- Check cases for the lookup' function
    print $ lookup' 4 myHeap    -- Expect 1
    print $ lookup' 10 myHeap   -- Expect 0
    print $ lookup' 10 empty'   -- Expect 0

    -- Check cases for the maxElement' function
    print $ maxElement' myHeap  -- Expect Just 6
    print $ maxElement' $ fromList' [0, 0, 0, 0, 0] -- Expect Just 0

    -- Check cases for the delete' function
    print $ delete' 3 myHeap   -- Expect Branch (Just 1,(Just (Branch (Just 2,(Just (Leaf 5),Just (Leaf 6)))),Just (Leaf 4)))
    print $ delete' 10 myHeap   -- Expect Branch (Just 1,(Just (Branch (Just 3,(Just (Leaf 5),Just (Leaf 4)))),Just (Branch (Just 2,(Just (Leaf 6),Nothing)))))

    -- Check cases for the isValidMinHeap' function
    print $ isValidMinHeap' myHeap  -- Expect 1
    print $ isValidMinHeap' (Leaf 0)    -- Expect 1'
    print $ isValidMinHeap' $ Branch (Just 2, (Just (Leaf 3), Just (Leaf 1)))   -- Expect 0
    print $ isValidMinHeap' $ Branch (Just 1,(Just (Branch (Just 0,(Just (Leaf 5),Just (Leaf 4)))),Just (Branch (Just 2,(Just (Leaf 6),Nothing))))) -- Expect 0
    print $ isValidMinHeap' $ Branch (Just 2, (Nothing, Nothing))   -- Expect 0
    print $ isValidMinHeap' $ Branch (Just 2, (Nothing, Just (Leaf 3)))   -- Expect 0
    print $ isValidMinHeap' $ Branch (Just 2, (Just (Leaf 1), Nothing))   -- Expect 0