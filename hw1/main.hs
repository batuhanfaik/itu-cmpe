module Main where
import System.Environment(getArgs)
import Data.Char(intToDigit, digitToInt)

parseArgs :: [String] -> [Char]
parseArgs args
    | head args == "d2c" = digitsToChars (read $ args !! 1) (read $ args !! 2)
    | head args == "c2d" = charsToDigits (read $ args !! 1) (args !! 2)
    | otherwise = error "Unrecognized operation!"

digitsToChars :: Int -> Int -> [Char]
digitsToChars base dVal
    | base <= dVal = error "Invalid digit!"
    | otherwise = mapDecToHex dVal

mapDecToHex :: Int -> [Char]
mapDecToHex x
    | x < 10 = show $ intToDigit x
    | x == 10 = "A"
    | x == 11 = "B"
    | x == 12 = "C"
    | x == 13 = "D"
    | x == 14 = "E"
    | x == 15 = "F"

charsToDigits :: Int -> [Char] -> [Char]
charsToDigits base cVal
    | base <= read (mapHexToDec cVal) = error "Invalid digit!"
    | otherwise = mapHexToDec cVal

mapHexToDec :: [Char] -> [Char]
mapHexToDec x
    | x == "A" = "10"
    | x == "B" = "11"
    | x == "C" = "12"
    | x == "D" = "13"
    | x == "E" = "14"
    | x == "F" = "15"
    | otherwise = x

main :: IO ()
main = do
    args <- getArgs
    putStrLn $ parseArgs args