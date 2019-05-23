-- Every string is reversed so we can work with the beginning of lists
-- E.g., lrlr suffix normally is rlrl prefix here

-- forbidden -> str -> bool
checkForbiddenPrefix :: Eq a => [a] -> [a] -> Bool
checkForbiddenPrefix = flip =<< ((==) .) . take . length

-- chars -> forbidden -> str -> [str]
buildValid :: Eq a => [[a]] -> [a] -> [a] -> [[a]]
buildValid chars forbidden str = filter (not . (checkForbiddenPrefix forbidden)) $ map (uncurry (++)) $ zip chars $ repeat str

flatten :: Foldable t => t [a] -> [a]
flatten = foldl (++) []

myValid :: [Char] -> [[Char]]
myValid = buildValid ["l","r","f"] "lrlr"

trajectories = iterate (flatten . map myValid) [""]
