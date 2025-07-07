// get all drivers by state, city filter
import {NextRequest, NextResponse} from "next/server";
import dbConnect from "@/lib/db";
import {Driver} from "@/model/driver";
import {getServerSession, User} from "next-auth";
import {authOptions} from "@/app/api/auth/[...nextauth]/options";

export async function GET(req: NextRequest) {
    try {
        await dbConnect();
        const session = await getServerSession(authOptions);
        const user: User = session?.user as User;

        if (!session || !user) {
            return NextResponse.json(
                { message: "User not logged in." },
                { status: 401 }
            )
        }

        if (user.role !== "manager") {
            return NextResponse.json(
                { message: "User is not manager." },
                { status: 403 }
            )
        }

        const { city, state } = await req.json();

        if (!city) {
            return NextResponse.json(
                { message: "No city found" },
                { status: 400 }
            )
        }

        if (!state) {
            return NextResponse.json(
                { message: "No state found" },
                { status: 400 }
            );
        }

        const drivers = await Driver.aggregate([
            {
                $match: {
                    city: city,
                    state: state,
                    manager: null
                }
            },
            {
                $lookup: {
                    from: "users",
                    localField: "userId",
                    foreignField: "_id",
                    as: "user",
                    pipeline: [
                        {
                            $project: {
                                firstName: 1,
                                lastName: 1,
                                profile: 1,
                                email: 1,
                            }
                        }
                    ]
                }
            },
            {
                $unwind: "$user",
            }
        ])

        if (!drivers) {
            return NextResponse.json(
                { message: "Failed to find drivers" },
                { status: 500 }
            )
        }

        return NextResponse.json(
            drivers,
            { status: 200 }
        )
    } catch (e) {
        return NextResponse.json({ error: e }, { status: 500 });
    }
}