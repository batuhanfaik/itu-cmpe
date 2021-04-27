module Main where
import System.Environment(getArgs)
import Data.Char(intToDigit, digitToInt)

-- checkArgs a b (c:cs) 
checkArgs :: [[Char]] -> [Char]
checkArgs [] = []
checkArgs _ = error "At least three input arguments required"

parseArgs :: [Char] -> Int -> Int -> Char
parseArgs mode base vals
    | mode == "d2c" = digitsToChars base vals
    | otherwise = error "Other"

digitsToChars :: Int -> Int -> Char
digitsToChars base dVal
    | base <= dVal = error "Invalid digit!"
    | otherwise = mapDecToHex dVal

mapDecToHex :: Int -> Char
mapDecToHex x
    | x < 10 = intToDigit x
    | x == 10 = 'A'
    | x == 11 = 'B'
    | x == 12 = 'C'
    | x == 13 = 'D'
    | x == 14 = 'E'
    | x == 15 = 'F'

-- intToDigit x = show x

-- charsToDigits [mode, base, vals] = putStrLn (mode ++ " " ++ base ++ " " ++ vals)

main = do
    args <- getArgs
    checkArgs args

-- main = do
--     putStrLn [digitsToChars 16 14]