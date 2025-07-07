import { NextAuthOptions } from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';
import bcrypt from 'bcryptjs';
import dbConnect from '@/lib/db';
import { User } from '@/model/user';

export const authOptions: NextAuthOptions = {
    providers: [
        CredentialsProvider({
            id: 'credentials',
            name: 'Credentials',
            credentials: {
                email: { label: 'Email', type: 'text' },
                password: { label: 'Password', type: 'password' },
            },
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            async authorize(credentials: any): Promise<any> {
                await dbConnect();
                try {
                    const user = await User.findOne({
                       email: credentials.email,
                    });
                    if (!user) {
                        throw new Error('No user found with this email');
                    }

                    const isPasswordCorrect = await bcrypt.compare(
                        credentials.password,
                        user.password
                    );
                    if (isPasswordCorrect) {
                        return {
                            _id: user._id?.toString(),
                            firstName: user.firstName,
                            lastName: user.lastName,
                            role: user.role
                        };
                    } else {
                        throw new Error('Incorrect password');
                    }
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                } catch (err: any) {
                    throw new Error(err);
                }
            },
        }),
    ],
    callbacks: {
        async jwt({ token, user }) {
            if (user) {
                token._id = user._id; // User ID
                token.firsName = user.firstName;
                token.lastName = user.lastName;
                token.role = user.role;
            }
            return token;
        },
        async session({ session, token }) {
            if (token) {
                session.user._id = token._id;
                session.user.firstName = token.firstName;
                session.user.lastName = token.lastName;
                session.user.role = token.role;
            }
            return session;
        },
    },
    session: {
        strategy: 'jwt',
    },
    secret: process.env.NEXTAUTH_SECRET,
    pages: {
        signIn: '/sign-in',
    },
    debug: true,
};
