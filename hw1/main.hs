module Main where
import System.Environment(getArgs)
import Data.Char(intToDigit, digitToInt)

parseArgs :: [String] -> Char
parseArgs args
    | head args == "d2c" = digitsToChars (read (args !! 1)) (read (args !! 2))
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

main = do
    args <- getArgs
    putStrLn [parseArgs args]